#include "hbm_read_protocol.h"
#include "offload_client.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace kvaf::hbm_read {
namespace {
addrinfo* Addresses(const std::string& address, bool listen) {
  auto split=address.rfind(':');
  if (split==std::string::npos || split==0 || split+1==address.size()) return nullptr;
  addrinfo hints{}, *list=nullptr;
  hints.ai_socktype=SOCK_STREAM; hints.ai_family=AF_INET; hints.ai_flags=listen?AI_PASSIVE:0;
  if (getaddrinfo(address.substr(0,split).c_str(),address.substr(split+1).c_str(),&hints,&list)) return nullptr;
  return list;
}
}
int Connect(const std::string& address) {
  auto* list=Addresses(address,false); if(!list) return -1;
  int fd=-1;
  for(auto* a=list;a;a=a->ai_next) {
    fd=socket(a->ai_family,a->ai_socktype,a->ai_protocol); if(fd<0) continue;
    int flags=fcntl(fd,F_GETFL,0);
    if(flags<0 || fcntl(fd,F_SETFL,flags|O_NONBLOCK)) { close(fd); fd=-1; continue; }
    int rc=connect(fd,a->ai_addr,a->ai_addrlen);
    if(rc<0 && errno==EINPROGRESS) {
      pollfd p{fd,POLLOUT,0}; int error=0; socklen_t size=sizeof(error);
      rc=poll(&p,1,5000)>0 && !getsockopt(fd,SOL_SOCKET,SO_ERROR,&error,&size) && !error ? 0:-1;
    }
    if(!rc && !fcntl(fd,F_SETFL,flags)) break;
    close(fd); fd=-1;
  }
  freeaddrinfo(list);
  if(fd>=0) {
    offload::SetSocketTimeout(fd);
    int yes=1;setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&yes,sizeof(yes));
  }
  return fd;
}
Status Listen(const std::string& address,int* fd,std::uint16_t* port) {
  auto* list=Addresses(address,true); if(!list) return Status::InvalidArgument("HBM listen requires IPv4 host:port");
  *fd=-1;
  for(auto* a=list;a;a=a->ai_next) {
    int s=socket(a->ai_family,a->ai_socktype,a->ai_protocol); if(s<0) continue;
    int yes=1; setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    if(!bind(s,a->ai_addr,a->ai_addrlen) && !listen(s,16)) { *fd=s; break; }
    close(s);
  }
  freeaddrinfo(list);
  if(*fd<0) return Status::Internal("cannot bind HBM control listener");
  sockaddr_in a{}; socklen_t n=sizeof(a);
  if(getsockname(*fd,reinterpret_cast<sockaddr*>(&a),&n)) {close(*fd);*fd=-1;return Status::Internal("HBM getsockname failed");}
  *port=ntohs(a.sin_port); return Status::OK();
}
ControlChannel::~ControlChannel() { Close(); }
void ControlChannel::Close() { if(fd_>=0) {close(fd_);fd_=-1;} }
bool ControlChannel::connected() {
  std::lock_guard<std::mutex> lock(mutex_);
  if(fd_<0)return false;
  pollfd p{fd_,POLLIN,0};
  if(poll(&p,1,0)>0 && p.revents) {
    char byte;
    if((p.revents&(POLLHUP|POLLERR|POLLNVAL)) || recv(fd_,&byte,1,MSG_PEEK|MSG_DONTWAIT)==0)Close();
  }
  return fd_>=0;
}
Status ControlChannel::Replace(int fd) {
  if(connected())return Status::Busy("HBM control connection is still live");
  std::lock_guard<std::mutex> lock(mutex_);fd_=fd;return Status::OK();
}
Status ControlChannel::Exchange(Request r,Response* out) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(fd_<0)return Status::InDoubt("HBM control disconnected; wait for producer reconnect, then resolve");
  r.count=0;r.header.bytes=RequestBytes(r.header.command);
  *out={};
  const auto response_bytes=ResponseBytes(r.header.command);
  if(!offload::WriteBytes(fd_,&r,r.header.bytes) || !offload::ReadBytes(fd_,&out->header,sizeof(out->header)) ||
     out->header.magic!=kMagic || out->header.version!=kVersion || out->header.bytes!=response_bytes ||
     !offload::ReadBytes(fd_,reinterpret_cast<char*>(out)+sizeof(out->header),response_bytes-sizeof(out->header))) {
    Close();return Status::InDoubt("HBM control response lost; retain source/output and resolve");
  }
  if(out->header.magic!=kMagic || out->header.version!=kVersion || out->header.bytes!=response_bytes ||
      out->header.command!=(r.header.command|0x8000U) || out->source_id!=r.source_id || out->epoch!=r.epoch ||
      out->client!=r.client || out->sequence!=r.sequence || out->mr_id!=r.mr_id || out->code>unsigned(StatusCode::kInDoubt)) {
    Close();return Status::InDoubt("HBM response identity/version mismatch");
  }
  return Status::Error(static_cast<StatusCode>(out->code),std::string(out->detail,strnlen(out->detail,sizeof(out->detail))));
}
Status SendHello(int fd,Admission command,const HbmReceiverTicket& receiver,const HbmReadTicket& ticket) {
  Hello h{};h.header={kMagic,kVersion,std::uint16_t(command),std::uint32_t(sizeof(h)+ticket.mr_count()*8)};
  h.receiver_id=receiver.receiver_id;h.receiver_epoch=receiver.receiver_epoch;
  h.source_id=ticket.source_id;h.source_epoch=ticket.epoch;h.token=ticket.token;h.mr_count=ticket.mr_count();
  const auto& sizes=ticket.mr_bytes;
  return offload::WriteBytes(fd,&h,sizeof(h)) && offload::WriteBytes(fd,sizes.data(),sizes.size()*8)?Status::OK():Status::InDoubt("HBM admission send failed");
}
Status ReadHello(int fd,Hello* h,HbmReadTicket* ticket) {
  if(!offload::ReadBytes(fd,h,sizeof(*h)) || h->header.magic!=kMagic || h->header.version!=kVersion ||
     (h->header.command!=unsigned(Admission::Join) && h->header.command!=unsigned(Admission::Leave) && h->header.command!=unsigned(Admission::Resume)) ||
     !h->mr_count || h->mr_count>4096 || h->header.bytes!=sizeof(*h)+h->mr_count*8)
    return Status::InvalidArgument("invalid HBM admission/version");
  std::vector<std::uint64_t> sizes(h->mr_count);
  if(!offload::ReadBytes(fd,sizes.data(),sizes.size()*8))return Status::InDoubt("HBM admission truncated");
  *ticket={h->source_id,h->source_epoch,h->token,std::move(sizes)};
  return ticket->valid()?Status::OK():Status::InvalidArgument("invalid HBM source manifest");
}
bool SendResult(int fd,const Status& s) {
  HelloResult r{};r.header={kMagic,kVersion,unsigned(Admission::Result),sizeof(r)};r.code=unsigned(s.code());
  std::snprintf(r.detail,sizeof(r.detail),"%s",s.message().c_str());return offload::WriteBytes(fd,&r,sizeof(r));
}
Status ReadResult(int fd,const peer::MessageHeader& header) {
  HelloResult r{};r.header=header;
  if(header.magic!=kMagic || header.version!=kVersion || header.command!=unsigned(Admission::Result) || header.bytes!=sizeof(r) ||
     !offload::ReadBytes(fd,reinterpret_cast<char*>(&r)+sizeof(header),sizeof(r)-sizeof(header)) || r.code>unsigned(StatusCode::kInDoubt))
    return Status::InDoubt("HBM admission response lost or invalid");
  return Status::Error(static_cast<StatusCode>(r.code),std::string(r.detail,strnlen(r.detail,sizeof(r.detail))));
}
Status Leave(const HbmReceiverTicket& receiver,const HbmReadTicket& ticket) {
  int fd=Connect(receiver.control_address);if(fd<0)return Status::InDoubt("HBM receiver unavailable; retain registration");
  auto status=SendHello(fd,Admission::Leave,receiver,ticket);
  peer::MessageHeader h{};
  if(status.ok())status=offload::ReadBytes(fd,&h,sizeof(h))?ReadResult(fd,h):Status::InDoubt("HBM detach acknowledgement lost");
  close(fd);return status;
}
}
