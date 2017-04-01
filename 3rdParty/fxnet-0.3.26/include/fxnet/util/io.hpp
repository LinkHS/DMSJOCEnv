#ifndef FXNET_UTIL_IO_H_
#define FXNET_UTIL_IO_H_

#ifdef WINDOWS

#include <io.h>
#include <process.h> /* for getpid() and the exec..() family */
#include <direct.h> /* for _getcwd() and _chdir() */
#else
#include <unistd.h>
#endif
#include <string>

#ifdef PROTOBUF_FULL
#include "google/protobuf/message.h"
#else
#include "google/protobuf/message_lite.h"
#endif

#include "fxnet/blob.hpp"
#include "fxnet/common.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

#ifdef PROTOBUF_FULL
using ::google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  bool res = ReadProtoFromTextFile(filename, proto);
  assert_force(res);
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const string filename);

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  bool res = ReadProtoFromBinaryFile(filename, proto);
  assert_force(res);
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const string filename);

bool ReadProtoFromCharArray(const char* buffer, int size, Message* proto);

inline void ReadProtoFromCharArrayOrDie(const char* buffer, int size, Message* proto){
	bool res = ReadProtoFromCharArray(buffer,size, proto);
	assert_force(res);
}

bool DecodeProtoFromFile(const string& filename, Message* proto, Message* param);


#else
using ::google::protobuf::MessageLite;
// ################ added by alan  start #######################
bool ReadProtoFromCharArray(const char* buffer, int size, MessageLite* proto);

inline void ReadProtoFromCharArrayOrDie(const char* buffer, int size, MessageLite* proto){
	bool res = ReadProtoFromCharArray(buffer,size, proto);
	assert_force(res);
}

bool DecodeProtoFromFile(const string& filename, MessageLite* proto, MessageLite* param);
void WriteProtoToBinaryFile(const MessageLite& proto, const string filename);
// ################ added by alan  end #######################
#endif

/*
inline void MakeTempFilename(string* temp_filename) {
  temp_filename->clear();
  *temp_filename = "/tmp/fxnet_test.XXXXXX";
  char* temp_filename_cstr = new char[temp_filename->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_filename_cstr, temp_filename->c_str());
  int fd = mkstemp(temp_filename_cstr);
  DBG(if(fd >= 0) std::cout<<"Failed to open a temporary file at: " << *temp_filename<<std::endl);
  assert_force(fd >= 0); // << "Failed to open a temporary file at: " << *temp_filename;
  close(fd);
  *temp_filename = temp_filename_cstr;
  delete[] temp_filename_cstr;
}

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  *temp_dirname = "/tmp/fxnet_test.XXXXXX";
  char* temp_dirname_cstr = new char[temp_dirname->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_dirname_cstr, temp_dirname->c_str());
  char* mkdtemp_result = mkdtemp(temp_dirname_cstr);
  DBG(if(!(mkdtemp_result != NULL)) std::cout<<"Failed to create a temporary directory at: "
		  << *temp_dirname<<std::endl;)
  assert_force(mkdtemp_result != NULL);
//      << "Failed to create a temporary directory at: " << *temp_dirname;
  *temp_dirname = temp_dirname_cstr;
  delete[] temp_dirname_cstr;
}

*/

bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);


}  // namespace fxnet
}  // namespace hbot
#endif   // FXNET_UTIL_IO_H_
