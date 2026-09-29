// Wire-compatible subset of the ONNX ModelProto schema. Field numbers and wire
// types come from https://github.com/onnx/onnx/blob/main/onnx/onnx.proto.
// Unknown protobuf fields are skipped. The importer deliberately exposes only
// graph structure and tensor metadata; it never treats this as an ONNX runtime.
export const ONNX_SCHEMA = `
syntax = "proto3";
package onnx;

message StringStringEntryProto { string key = 1; string value = 2; }
message OperatorSetIdProto { string domain = 1; int64 version = 2; }
message TensorShapeProto {
  message Dimension { oneof value { int64 dim_value = 1; string dim_param = 2; } }
  repeated Dimension dim = 1;
}
message TypeProto {
  message Tensor { int32 elem_type = 1; TensorShapeProto shape = 2; }
  oneof value { Tensor tensor_type = 1; }
}
message ValueInfoProto { string name = 1; TypeProto type = 2; string doc_string = 3; }
message TensorProto {
  repeated int64 dims = 1;
  int32 data_type = 2;
  repeated float float_data = 4 [packed = true];
  repeated int32 int32_data = 5 [packed = true];
  repeated bytes string_data = 6;
  repeated int64 int64_data = 7 [packed = true];
  string name = 8;
  bytes raw_data = 9;
  repeated double double_data = 10 [packed = true];
  repeated uint64 uint64_data = 11 [packed = true];
  repeated StringStringEntryProto external_data = 13;
  int32 data_location = 14;
}
message AttributeProto {
  string name = 1;
  float f = 2;
  int64 i = 3;
  bytes s = 4;
  TensorProto t = 5;
  GraphProto g = 6;
  repeated float floats = 7 [packed = true];
  repeated int64 ints = 8 [packed = true];
  repeated bytes strings = 9;
  repeated TensorProto tensors = 10;
  repeated GraphProto graphs = 11;
  int32 type = 20;
  string ref_attr_name = 21;
}
message NodeProto {
  repeated string input = 1;
  repeated string output = 2;
  string name = 3;
  string op_type = 4;
  repeated AttributeProto attribute = 5;
  string doc_string = 6;
  string domain = 7;
}
message GraphProto {
  repeated NodeProto node = 1;
  string name = 2;
  repeated TensorProto initializer = 5;
  repeated ValueInfoProto input = 11;
  repeated ValueInfoProto output = 12;
  repeated ValueInfoProto value_info = 13;
}
message ModelProto {
  int64 ir_version = 1;
  string producer_name = 2;
  string producer_version = 3;
  string domain = 4;
  int64 model_version = 5;
  string doc_string = 6;
  GraphProto graph = 7;
  repeated OperatorSetIdProto opset_import = 8;
  repeated StringStringEntryProto metadata_props = 14;
}
`;
