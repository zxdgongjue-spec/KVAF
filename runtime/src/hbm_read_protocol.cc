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
Status Exchange(const std::string& address,Request r,const std::vector<Span>& spans,Response* out,int* connection) {
  if(!out || spans.size()>offload::kMaxSpans) return Status::InvalidArgument("invalid HBM exchange");
  int fd=connection && *connection>=0?*connection:Connect(address);
  if(fd<0) return Status::InDoubt("HBM control connection unavailable");
  if(connection)*connection=-1;
  r.count=spans.size(); r.header.bytes=sizeof(r)+spans.size()*sizeof(Span);
  bool ok=offload::WriteBytes(fd,&r,sizeof(r)) && offload::WriteBytes(fd,spans.data(),spans.size()*sizeof(Span)) &&
      offload::ReadBytes(fd,out,sizeof(*out));
  if(!ok) {close(fd);return Status::InDoubt("HBM control response lost; retain source/output and resolve");}
  if(out->header.magic!=kMagic || out->header.version!=kVersion || out->header.bytes!=sizeof(Response) ||
      out->header.command!=(r.header.command|0x8000U) || out->source_id!=r.source_id || out->epoch!=r.epoch ||
      out->client!=r.client || out->sequence!=r.sequence || out->code>unsigned(StatusCode::kInDoubt)) {
    close(fd);return Status::InDoubt("HBM response identity/version mismatch");
  }
  if(connection && out->reserved==1U)*connection=fd;
  else close(fd);
  return Status::Error(static_cast<StatusCode>(out->code), std::string(out->detail,strnlen(out->detail,sizeof(out->detail))));
}
}
