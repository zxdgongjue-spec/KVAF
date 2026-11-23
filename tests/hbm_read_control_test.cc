#include "hbm_read_protocol.h"
#include "offload_client.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace hr=kvaf::hbm_read;
namespace off=kvaf::offload;
using kvaf::StatusCode;
void Require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
struct Sockets {
  int fd[2]{};
  Sockets() { Require(!socketpair(AF_UNIX,SOCK_STREAM,0,fd),"socketpair");for(int s:fd)off::SetSocketTimeout(s); }
  ~Sockets() { for(int s:fd)if(s>=0)close(s); }
};
hr::Request MakeRequest(hr::Command command) {
  auto r=hr::Make(command,kvaf::HbmReadTicket{13,17,19,{4096}},23);
  r.sequence=29;r.connection.key[0][31]=0xa5;return r;
}
hr::Response MakeResponse(const hr::Request& r) {
  hr::Response response{};
  response.header={hr::kMagic,hr::kVersion,std::uint16_t(r.header.command|0x8000U),hr::ResponseBytes(r.header.command)};
  response.source_id=r.source_id;response.epoch=r.epoch;response.client=r.client;
  response.sequence=r.sequence;response.mr_id=r.mr_id;return response;
}
void FragmentedWrite(int fd,const void* frame,std::size_t bytes) {
  const auto* data=static_cast<const char*>(frame);
  for(std::size_t offset=0;offset<bytes;) {
    const auto n=std::min<std::size_t>(offset<12?5:37,bytes-offset);
    Require(off::WriteBytes(fd,data+offset,n),"fragment write");offset+=n;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
void SourceFrames() {
  Sockets sockets;
  // Buffer mixed frame kinds together, including Result followed immediately
  // by another request. A receiver must consume exactly one frame each time.
  std::vector<hr::Request> sent;
  for(auto command:{hr::Command::Memory,hr::Command::Open,hr::Command::Channel,hr::Command::Confirm}) {
    sent.push_back(MakeRequest(command));const auto& r=sent.back();
    Require(off::WriteBytes(sockets.fd[0],&r,r.header.bytes),"queue source frame");
  }
  hr::HelloResult admission{};
  admission.header={hr::kMagic,hr::kVersion,unsigned(hr::Admission::Result),sizeof(admission)};
  Require(off::WriteBytes(sockets.fd[0],&admission,sizeof(admission)),"queue admission result");
  auto last=MakeRequest(hr::Command::Memory);last.sequence++;
  Require(off::WriteBytes(sockets.fd[0],&last,last.header.bytes),"queue next frame");
  for(const auto& expected:sent) {
    hr::Request request{};hr::HelloResult result{};
    Require(hr::ReadSourceMessage(sockets.fd[1],&request,&result),"read source frame");
    Require(!std::memcmp(&expected,&request,expected.header.bytes),"source frame bytes preserved");
  }
  hr::Request request{};hr::HelloResult result{};
  Require(hr::ReadSourceMessage(sockets.fd[1],&request,&result) && hr::ReadResult(result).ok(),"decode admission result");
  Require(hr::ReadSourceMessage(sockets.fd[1],&request,&result) && request.sequence==last.sequence,"admission does not consume next request");
  for(auto command:{hr::Command::Memory,hr::Command::Open}) {
    const auto expected=MakeRequest(command);
    auto writer=std::async(std::launch::async,[&]{FragmentedWrite(sockets.fd[0],&expected,expected.header.bytes);});
    Require(hr::ReadSourceMessage(sockets.fd[1],&request,&result) &&
            !std::memcmp(&expected,&request,expected.header.bytes),"fragmented source frame");writer.get();
  }
  for(int fault=0;fault<4;++fault) {
    Sockets bad;auto r=MakeRequest(hr::Command::Memory);
    if(fault==0)r.header.magic++;
    if(fault==1)r.header.version--;
    if(fault==2)r.header.bytes++;
    if(fault==3)r.count=1;
    const auto bytes=fault==3?hr::RequestBytes(r.header.command):sizeof(r.header);
    Require(off::WriteBytes(bad.fd[0],&r,bytes),"invalid source frame");
    Require(!hr::ReadSourceMessage(bad.fd[1],&request,&result),"reject malformed header before waiting for body");
  }
  Sockets truncated;auto r=MakeRequest(hr::Command::Open);
  Require(off::WriteBytes(truncated.fd[0],&r,hr::RequestBytes(unsigned(hr::Command::Memory))),"short source frame");
  shutdown(truncated.fd[0],SHUT_WR);
  Require(!hr::ReadSourceMessage(truncated.fd[1],&request,&result),"reject truncated connection tail");
}
void Responses() {
  for(int fault=0;fault<9;++fault) {
    Sockets sockets;const auto request=MakeRequest(hr::Command::Memory);
    hr::ControlChannel client(sockets.fd[0]);sockets.fd[0]=-1;
    auto responder=std::async(std::launch::async,[&]{
      hr::Request received{};Require(off::ReadBytes(sockets.fd[1],&received,request.header.bytes),"receive request");
      auto response=MakeResponse(received);
      if(fault==1)response.sequence++;
      if(fault==2)response.mr_id++;
      if(fault==3)response.header.version--;
      if(fault==4)response.header.bytes++;
      if(fault==5)response.code=unsigned(StatusCode::kInDoubt)+1;
      if(fault==7)response.code=unsigned(StatusCode::kBusy);
      const auto bytes=fault==3 || fault==4?sizeof(response.header):response.header.bytes;
      if(fault==6) {
        Require(off::WriteBytes(sockets.fd[1],&response,17),"truncated reply");shutdown(sockets.fd[1],SHUT_WR);
      } else if(fault==8)FragmentedWrite(sockets.fd[1],&response,bytes);
      else Require(off::WriteBytes(sockets.fd[1],&response,bytes),"send reply");
    });
    hr::Response response{};const auto status=client.Exchange(request,&response);responder.get();
    if(fault==0 || fault==8)Require(status.ok() && client.connected(),"successful reply keeps channel");
    else if(fault==7)Require(status.code()==StatusCode::kBusy && client.connected(),"remote error is preserved");
    else Require(status.code()==StatusCode::kInDoubt && !client.connected(),"invalid/lost response closes TCP without claiming completion");
  }
}

int main() {try {
  SourceFrames();Responses();std::cout<<"HBM initialization/control framing and reply validation passed\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
