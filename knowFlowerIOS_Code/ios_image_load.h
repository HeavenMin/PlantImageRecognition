

#ifndef TENSORFLOW_CONTRIB_IOS_EXAMPLES_CAMERA_IMAGE_LOAD_H_
#define TENSORFLOW_CONTRIB_IOS_EXAMPLES_CAMERA_IMAGE_LOAD_H_

#include <vector>

#include "tensorflow/core/framework/types.h"

std::vector<tensorflow::uint8> LoadImageFromFile(const char* file_name,
                                                 int* out_width,
                                                 int* out_height,
                                                 int* out_channels);

#endif 
