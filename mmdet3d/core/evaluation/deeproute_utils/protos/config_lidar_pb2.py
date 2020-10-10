# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: config_lidar.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import protos.geometry_pb2 as geometry__pb2
import protos.stream_pb2 as stream__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='config_lidar.proto',
  package='deeproute.common',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x12\x63onfig_lidar.proto\x12\x10\x64\x65\x65proute.common\x1a\x0egeometry.proto\x1a\x0cstream.proto\"\x96\x01\n\tIntrinsic\x12\x15\n\x07\x65nabled\x18\x01 \x01(\x08:\x04true\x12\x19\n\x0e\x61zimuth_offset\x18\x03 \x01(\x02:\x01\x30\x12\x0f\n\x04vert\x18\x04 \x01(\x02:\x01\x30\x12\x16\n\x0bvert_offset\x18\x05 \x01(\x02:\x01\x30\x12\x19\n\x11laser_time_offset\x18\x06 \x01(\x03\x12\x13\n\x0blaser_index\x18\t \x01(\r\"t\n\x0e\x43\x61\x63heIntrinsic\x12\x0f\n\x07\x63os_rot\x18\x01 \x01(\x02\x12\x0f\n\x07sin_rot\x18\x02 \x01(\x02\x12\x10\n\x08\x63os_vert\x18\x03 \x01(\x02\x12\x10\n\x08sin_vert\x18\x04 \x01(\x02\x12\x1c\n\x14\x66inal_azimuth_offset\x18\x05 \x01(\x02\"\xc0\x02\n\tModelSpec\x12$\n\x04type\x18\x01 \x02(\x0e\x32\x16.deeproute.common.Type\x12\x12\n\nlaser_size\x18\x02 \x02(\r\x12\x14\n\x0cmin_distance\x18\x03 \x01(\x02\x12\x14\n\x0cmax_distance\x18\x04 \x01(\x02\x12\x14\n\x0claser_period\x18\x05 \x01(\r\x12\x15\n\rfiring_period\x18\x06 \x01(\r\x12\x1a\n\x12\x64\x61ta_packet_period\x18\x07 \x01(\r\x12\x18\n\x10\x64\x61ta_packet_size\x18\x08 \x01(\r\x12\x1c\n\x14position_packet_size\x18\t \x01(\r\x12\x13\n\x0b\x62locks_size\x18\n \x01(\r\x12\x37\n\x12\x64\x65\x66\x61ult_intrinsics\x18\x0b \x03(\x0b\x32\x1b.deeproute.common.Intrinsic\"\xbd\x02\n\x05Lidar\x12\x0c\n\x04name\x18\x01 \x02(\x0c\x12$\n\x04type\x18\x02 \x02(\x0e\x32\x16.deeproute.common.Type\x12/\n\nintrinsics\x18\x03 \x03(\x0b\x32\x1b.deeproute.common.Intrinsic\x12)\n\x04spec\x18\x04 \x01(\x0b\x32\x1b.deeproute.common.ModelSpec\x12<\n\x10sensing_to_lidar\x18\x05 \x01(\x0b\x32\".deeproute.common.Transformation3f\x12(\n\x06stream\x18\x06 \x01(\x0b\x32\x18.deeproute.common.Stream\x12\x10\n\x03rpm\x18\x07 \x01(\x01:\x03\x36\x30\x30\x12\x14\n\x0cmin_distance\x18\x08 \x01(\x02\x12\x14\n\x0cmax_distance\x18\t \x01(\x02\"\x91\x02\n\nLidarArray\x12\'\n\x06lidars\x18\x01 \x03(\x0b\x32\x17.deeproute.common.Lidar\x12>\n\x12vehicle_to_sensing\x18\x02 \x01(\x0b\x32\".deeproute.common.Transformation3f\x12@\n\rexclusion_box\x18\x04 \x01(\x0b\x32).deeproute.common.LidarArray.ExclusionBox\x1aX\n\x0c\x45xclusionBox\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\x13\n\x05z_neg\x18\x04 \x01(\x02:\x04-100\x12\x12\n\x05z_pos\x18\x05 \x01(\x02:\x03\x31\x30\x30*\xdb\x01\n\x04Type\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\r\n\tHDL_64_S2\x10\x01\x12\r\n\tHDL_64_S3\x10\x02\x12\n\n\x06HDL_32\x10\x03\x12\n\n\x06VLP_16\x10\x04\x12\r\n\tVLP_16_HR\x10\x05\x12\n\n\x06VLP_32\x10\x06\x12\r\n\tPANDAR_40\x10\x0b\x12\x10\n\x0cPANDAR_40_AC\x10\x0c\x12\r\n\tPANDAR_64\x10\r\x12\x0c\n\x08RFANS_16\x10\x10\x12\x10\n\x0cRFANS_16_GPS\x10\x11\x12\x0c\n\x08RFANS_32\x10\x12\x12\t\n\x05RS_16\x10\x15\x12\x08\n\x04\x46PGA\x10\x16')
  ,
  dependencies=[geometry__pb2.DESCRIPTOR,stream__pb2.DESCRIPTOR,])

_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='deeproute.common.Type',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HDL_64_S2', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HDL_64_S3', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HDL_32', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VLP_16', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VLP_16_HR', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VLP_32', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PANDAR_40', index=7, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PANDAR_40_AC', index=8, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PANDAR_64', index=9, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RFANS_16', index=10, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RFANS_16_GPS', index=11, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RFANS_32', index=12, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RS_16', index=13, number=21,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FPGA', index=14, number=22,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1261,
  serialized_end=1480,
)
_sym_db.RegisterEnumDescriptor(_TYPE)

Type = enum_type_wrapper.EnumTypeWrapper(_TYPE)
UNSPECIFIED = 0
HDL_64_S2 = 1
HDL_64_S3 = 2
HDL_32 = 3
VLP_16 = 4
VLP_16_HR = 5
VLP_32 = 6
PANDAR_40 = 11
PANDAR_40_AC = 12
PANDAR_64 = 13
RFANS_16 = 16
RFANS_16_GPS = 17
RFANS_32 = 18
RS_16 = 21
FPGA = 22



_INTRINSIC = _descriptor.Descriptor(
  name='Intrinsic',
  full_name='deeproute.common.Intrinsic',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='enabled', full_name='deeproute.common.Intrinsic.enabled', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='azimuth_offset', full_name='deeproute.common.Intrinsic.azimuth_offset', index=1,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vert', full_name='deeproute.common.Intrinsic.vert', index=2,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vert_offset', full_name='deeproute.common.Intrinsic.vert_offset', index=3,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='laser_time_offset', full_name='deeproute.common.Intrinsic.laser_time_offset', index=4,
      number=6, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='laser_index', full_name='deeproute.common.Intrinsic.laser_index', index=5,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=71,
  serialized_end=221,
)


_CACHEINTRINSIC = _descriptor.Descriptor(
  name='CacheIntrinsic',
  full_name='deeproute.common.CacheIntrinsic',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cos_rot', full_name='deeproute.common.CacheIntrinsic.cos_rot', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sin_rot', full_name='deeproute.common.CacheIntrinsic.sin_rot', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cos_vert', full_name='deeproute.common.CacheIntrinsic.cos_vert', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sin_vert', full_name='deeproute.common.CacheIntrinsic.sin_vert', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='final_azimuth_offset', full_name='deeproute.common.CacheIntrinsic.final_azimuth_offset', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=223,
  serialized_end=339,
)


_MODELSPEC = _descriptor.Descriptor(
  name='ModelSpec',
  full_name='deeproute.common.ModelSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='deeproute.common.ModelSpec.type', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='laser_size', full_name='deeproute.common.ModelSpec.laser_size', index=1,
      number=2, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_distance', full_name='deeproute.common.ModelSpec.min_distance', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_distance', full_name='deeproute.common.ModelSpec.max_distance', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='laser_period', full_name='deeproute.common.ModelSpec.laser_period', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='firing_period', full_name='deeproute.common.ModelSpec.firing_period', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_packet_period', full_name='deeproute.common.ModelSpec.data_packet_period', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_packet_size', full_name='deeproute.common.ModelSpec.data_packet_size', index=7,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='position_packet_size', full_name='deeproute.common.ModelSpec.position_packet_size', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blocks_size', full_name='deeproute.common.ModelSpec.blocks_size', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='default_intrinsics', full_name='deeproute.common.ModelSpec.default_intrinsics', index=10,
      number=11, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=342,
  serialized_end=662,
)


_LIDAR = _descriptor.Descriptor(
  name='Lidar',
  full_name='deeproute.common.Lidar',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='deeproute.common.Lidar.name', index=0,
      number=1, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='deeproute.common.Lidar.type', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='intrinsics', full_name='deeproute.common.Lidar.intrinsics', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='spec', full_name='deeproute.common.Lidar.spec', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sensing_to_lidar', full_name='deeproute.common.Lidar.sensing_to_lidar', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stream', full_name='deeproute.common.Lidar.stream', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rpm', full_name='deeproute.common.Lidar.rpm', index=6,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=float(600),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_distance', full_name='deeproute.common.Lidar.min_distance', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_distance', full_name='deeproute.common.Lidar.max_distance', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=665,
  serialized_end=982,
)


_LIDARARRAY_EXCLUSIONBOX = _descriptor.Descriptor(
  name='ExclusionBox',
  full_name='deeproute.common.LidarArray.ExclusionBox',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='deeproute.common.LidarArray.ExclusionBox.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='deeproute.common.LidarArray.ExclusionBox.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='deeproute.common.LidarArray.ExclusionBox.z', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z_neg', full_name='deeproute.common.LidarArray.ExclusionBox.z_neg', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(-100),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z_pos', full_name='deeproute.common.LidarArray.ExclusionBox.z_pos', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(100),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1170,
  serialized_end=1258,
)

_LIDARARRAY = _descriptor.Descriptor(
  name='LidarArray',
  full_name='deeproute.common.LidarArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lidars', full_name='deeproute.common.LidarArray.lidars', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vehicle_to_sensing', full_name='deeproute.common.LidarArray.vehicle_to_sensing', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='exclusion_box', full_name='deeproute.common.LidarArray.exclusion_box', index=2,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_LIDARARRAY_EXCLUSIONBOX, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=985,
  serialized_end=1258,
)

_MODELSPEC.fields_by_name['type'].enum_type = _TYPE
_MODELSPEC.fields_by_name['default_intrinsics'].message_type = _INTRINSIC
_LIDAR.fields_by_name['type'].enum_type = _TYPE
_LIDAR.fields_by_name['intrinsics'].message_type = _INTRINSIC
_LIDAR.fields_by_name['spec'].message_type = _MODELSPEC
_LIDAR.fields_by_name['sensing_to_lidar'].message_type = geometry__pb2._TRANSFORMATION3F
_LIDAR.fields_by_name['stream'].message_type = stream__pb2._STREAM
_LIDARARRAY_EXCLUSIONBOX.containing_type = _LIDARARRAY
_LIDARARRAY.fields_by_name['lidars'].message_type = _LIDAR
_LIDARARRAY.fields_by_name['vehicle_to_sensing'].message_type = geometry__pb2._TRANSFORMATION3F
_LIDARARRAY.fields_by_name['exclusion_box'].message_type = _LIDARARRAY_EXCLUSIONBOX
DESCRIPTOR.message_types_by_name['Intrinsic'] = _INTRINSIC
DESCRIPTOR.message_types_by_name['CacheIntrinsic'] = _CACHEINTRINSIC
DESCRIPTOR.message_types_by_name['ModelSpec'] = _MODELSPEC
DESCRIPTOR.message_types_by_name['Lidar'] = _LIDAR
DESCRIPTOR.message_types_by_name['LidarArray'] = _LIDARARRAY
DESCRIPTOR.enum_types_by_name['Type'] = _TYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Intrinsic = _reflection.GeneratedProtocolMessageType('Intrinsic', (_message.Message,), {
  'DESCRIPTOR' : _INTRINSIC,
  '__module__' : 'config_lidar_pb2'
  # @@protoc_insertion_point(class_scope:deeproute.common.Intrinsic)
  })
_sym_db.RegisterMessage(Intrinsic)

CacheIntrinsic = _reflection.GeneratedProtocolMessageType('CacheIntrinsic', (_message.Message,), {
  'DESCRIPTOR' : _CACHEINTRINSIC,
  '__module__' : 'config_lidar_pb2'
  # @@protoc_insertion_point(class_scope:deeproute.common.CacheIntrinsic)
  })
_sym_db.RegisterMessage(CacheIntrinsic)

ModelSpec = _reflection.GeneratedProtocolMessageType('ModelSpec', (_message.Message,), {
  'DESCRIPTOR' : _MODELSPEC,
  '__module__' : 'config_lidar_pb2'
  # @@protoc_insertion_point(class_scope:deeproute.common.ModelSpec)
  })
_sym_db.RegisterMessage(ModelSpec)

Lidar = _reflection.GeneratedProtocolMessageType('Lidar', (_message.Message,), {
  'DESCRIPTOR' : _LIDAR,
  '__module__' : 'config_lidar_pb2'
  # @@protoc_insertion_point(class_scope:deeproute.common.Lidar)
  })
_sym_db.RegisterMessage(Lidar)

LidarArray = _reflection.GeneratedProtocolMessageType('LidarArray', (_message.Message,), {

  'ExclusionBox' : _reflection.GeneratedProtocolMessageType('ExclusionBox', (_message.Message,), {
    'DESCRIPTOR' : _LIDARARRAY_EXCLUSIONBOX,
    '__module__' : 'config_lidar_pb2'
    # @@protoc_insertion_point(class_scope:deeproute.common.LidarArray.ExclusionBox)
    })
  ,
  'DESCRIPTOR' : _LIDARARRAY,
  '__module__' : 'config_lidar_pb2'
  # @@protoc_insertion_point(class_scope:deeproute.common.LidarArray)
  })
_sym_db.RegisterMessage(LidarArray)
_sym_db.RegisterMessage(LidarArray.ExclusionBox)


# @@protoc_insertion_point(module_scope)
