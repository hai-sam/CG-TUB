#include <Eigen/Dense>
#include <_stdio.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <ostream>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <sstream>
#include <string>
#include "args/args.hxx"
#include "imgui.h"
#include "polyscope/combining_hash_functions.h"
#include "polyscope/curve_network.h"
#include "polyscope/file_helpers.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "portable-file-dialogs.h"

int edgeTable[256]={
0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };
int triTable[256][16] =
{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

using Point = std::array<float, 3>;
using Normal = std::array<float, 3>;

typedef struct {
  Point p[3];
} Triangle;

void readOff(const std::string& filename, std::vector<Point> &points,
             std::vector<Normal> &normals) {
  points.clear();
  normals.clear();

  std::string line;
  std::ifstream file(filename);

  std::getline(file, line);
  std::istringstream formatStream(line);
  std::string format;
  formatStream >> format;
  std::getline(file, line);

  std::istringstream meta(
      line); // read vertices and faces idk if its needed | edges isnt used
  int Nvertices, Nfaces, Nedges;
  meta >> Nvertices >> Nfaces >> Nedges;
  if (format == "OFF") {
    for (int i = 0; i < Nvertices; i++) {
      std::getline(file, line);
      std::istringstream pointLine(line);
      float x, y, z;
      pointLine >> x >> y >> z;

      points.emplace_back(Point{x, y, z});
    }
  }else if (format == "NOFF") {
    for (int i = 0; i < Nvertices; i++) {
      std::getline(file, line);
      std::istringstream pointLine(line);
      float x, y, z, nx, ny, nz;
      pointLine >> x >> y >> z >> nx >> ny >> nz;

      points.emplace_back(Point{x, y, z});
      normals.emplace_back(Normal{nx,ny,nz});
    }
  }else {
    return;
  }
  file.close();
}

void readOff(const std::string &filename, std::vector<Point> &points) {
  std::vector<Normal> dummy_normals;
  readOff(filename, points, dummy_normals); // reuse the full version
}

struct EuclideanDistance {
  static float measure(Point const &p1, Point const &p2) {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    float dz = p1[2] - p2[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }
};

/*
 * This is not yet a spatial data structure :)
 */
class SpatialDataStructure {
public:
  SpatialDataStructure(std::vector<Point> const &points) : m_points(points) {}

  virtual ~SpatialDataStructure() = default;

  std::vector<Point> const &getPoints() const { return m_points; }

  virtual std::vector<std::size_t> collectInRadius(Point const &p,
                                                   float radius) const {
    std::vector<std::size_t> result;

    // Dummy brute-force implementation
    // Implemented in kdTree subclass
    for (std::size_t i = 0; i < m_points.size(); ++i) {
      float distance = EuclideanDistance::measure(p, m_points[i]);
      if (distance <= radius)
        result.push_back(i);
    }

    return result;
  }

  virtual std::vector<std::size_t> collectKNearest(Point const &p,
                                                   unsigned int k) const {

    std::vector<std::size_t> result;

    // Bogus knn implementation, giving you the first k points!
    // Implemented in kdTree subclass
    for (std::size_t i = 0; (i < k) && (i < m_points.size()); ++i) {
      result.push_back(i);
    }

    return result;
  }

private:
  std::vector<Point> m_points;
};

struct Node { // to represent the Nodes in the kd-tree (if both child nodes are
              // null -> leaf node)
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
  std::vector<std::size_t> indices;
  int pointIndex;
  bool isLeaf;
  int splitAxis;
};

class KDTree : public SpatialDataStructure {
private:
  std::unique_ptr<Node> root;

public:
  KDTree(std::vector<Point> const &points) : SpatialDataStructure(points) {
    std::vector<std::size_t> indices(points.size());
    for (std::size_t i = 0; i < points.size(); i++)
      indices[i] = i;
    root = BuildTree(indices, 0);
  }
  std::unique_ptr<Node> BuildTree(std::vector<std::size_t> indices, int axis) {
    if (indices.size() == 0)
      return nullptr;
    Node n;
    if (indices.size() <= 1) {
      n.isLeaf = true;
      n.pointIndex = indices[0];
      n.indices = indices;
      return std::make_unique<Node>(std::move(n));
    }
    n.splitAxis = axis;
    n.indices = indices;
    std::vector<std::size_t> left;
    std::vector<std::size_t> right;

    for (unsigned int i = 0; i < indices.size(); i++) {
      if (i == indices.size() / 2) {
        n.pointIndex = indices[indices.size() / 2];
      } else if (getPoints()[indices[i]][axis] <
                 getPoints()[indices[indices.size() / 2]][axis]) {
        left.emplace_back(indices[i]);
      } else {
        right.emplace_back(indices[i]);
      }
    }
    n.isLeaf = false;
    if (left.size()) {
      n.left = BuildTree(left, (axis + 1) % 3);
    }
    if (right.size()) {
      n.right = BuildTree(right, (axis + 1) % 3);
    }
    return std::make_unique<Node>(std::move(n));
  }
  std::vector<std::size_t> collectKNearest(Point const &p,
                                           unsigned int k) const {
    std::vector<std::size_t> indices;
    std::priority_queue<std::pair<float, std::size_t>> pq;
    SearchTreeK(root.get(), p, pq, k);
    while (!pq.empty()) {
      indices.push_back(pq.top().second);
      pq.pop();
    }
    return indices;
  }
  std::vector<std::size_t> collectInRadius(Point const &p, float radius) const {
    std::vector<std::size_t> indices;

    SearchTreeRadius(root.get(), p, indices, radius);

    return indices;
  }

  // ex_2 implement 2d distance search using the tree (no z-coordinates) - only
  // x and y
  std::vector<std::size_t> collectInRadius2D(Point const &p,
                                             float radius) const {
    std::vector<std::size_t> indices;
    SearchTreeRadius2D(root.get(), p, indices, radius);
    return indices;
  }

  void SearchTreeRadius2D(Node *n, Point const &p, std::vector<size_t> &result,
                          float radius) const {
    if (n == nullptr)
      return;

    float radiusSq = radius * radius;

    if (n->isLeaf) {
      float dx = p[0] - getPoints()[n->pointIndex][0];
      float dy = p[1] - getPoints()[n->pointIndex][1];
      float distSq = dx * dx + dy * dy;
      if (distSq < radiusSq)
        result.push_back(n->pointIndex);
      return;
    }
    float dx = p[0] - getPoints()[n->pointIndex][0];
    float dy = p[1] - getPoints()[n->pointIndex][1];
    float distSq = dx * dx + dy * dy;
    if (distSq < radiusSq) {
      result.push_back(n->pointIndex);
    }
    if (n->splitAxis == 2) {
      SearchTreeRadius2D(n->left.get(), p, result, radius);
      SearchTreeRadius2D(n->right.get(), p, result, radius);
    } else {
      // check if radius touches the plane
      if (p[n->splitAxis] < getPoints()[n->pointIndex][n->splitAxis]) {
        SearchTreeRadius2D(n->left.get(), p, result, radius);

        float planeDistance =
            abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);

        if (planeDistance * planeDistance < radiusSq) {
          SearchTreeRadius2D(n->right.get(), p, result, radius);
        }
      } else {
        SearchTreeRadius2D(n->right.get(), p, result, radius);
        float planeDistance =
            abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
        if (planeDistance * planeDistance < radiusSq) {
          SearchTreeRadius2D(n->left.get(), p, result, radius);
        }
      }
    }
  }

  void SearchTreeRadius(Node *n, Point const &p, std::vector<size_t> &result,
                        float radius) const {
    if (n == nullptr)
      return;
    if (n->isLeaf) {
      float dist = EuclideanDistance::measure(p, getPoints()[n->pointIndex]);
      if (dist < radius)
        result.push_back(n->pointIndex);
      return;
    }
    float dist = EuclideanDistance::measure(p, getPoints()[n->pointIndex]);
    if (dist < radius) {
      result.push_back(n->pointIndex);
    }
    if (p[n->splitAxis] < getPoints()[n->pointIndex][n->splitAxis]) {
      SearchTreeRadius(n->left.get(), p, result, radius);
      float planeDistance =
          abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
      if (planeDistance < radius) {
        SearchTreeRadius(n->right.get(), p, result, radius);
      }
    } else {
      SearchTreeRadius(n->right.get(), p, result, radius);
      float planeDistance =
          abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
      if (planeDistance < radius) {
        SearchTreeRadius(n->left.get(), p, result, radius);
      }
    }
  }
  void SearchTreeK(Node *n, Point const &p,
                   std::priority_queue<std::pair<float, std::size_t>> &result,
                   unsigned int k) const {
    if (n == nullptr)
      return;
    if (n->isLeaf) {
      float dist = EuclideanDistance::measure(p, getPoints()[n->pointIndex]);
      if (result.size() < k || dist < result.top().first) {
        result.push({dist, n->pointIndex});
        if (result.size() > k)
          result.pop(); // remove farthest
      }
      return;
    }
    float dist = EuclideanDistance::measure(p, getPoints()[n->pointIndex]);
    if (result.size() < k || dist < result.top().first) {
      result.push({dist, n->pointIndex});
      if (result.size() > k)
        result.pop(); // remove farthest
    }
    if (p[n->splitAxis] < getPoints()[n->pointIndex][n->splitAxis]) {
      SearchTreeK(n->left.get(), p, result, k);
      float planeDistance =
          abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
      if (result.size() < k || planeDistance < result.top().first) {
        SearchTreeK(n->right.get(), p, result, k);
      }
    } else {
      SearchTreeK(n->right.get(), p, result, k);
      float planeDistance =
          abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
      if (result.size() < k || planeDistance < result.top().first) {
        SearchTreeK(n->left.get(), p, result, k);
      }
    }
  }
  void visualiseBBoxes(Point bboxMin, Point bboxMax,
                       std::vector<Point> &vertices,
                       std::vector<std::array<int, 3>> &faces,
                       unsigned int bucketSize, int maxDepth) {
    visualiseBBoxes(root.get(), bboxMin, bboxMax, vertices, faces, bucketSize,
                    maxDepth);
  }

  void visualiseBBoxes(Node *n, Point bboxMin, Point bboxMax,
                       std::vector<Point> &vertices,
                       std::vector<std::array<int, 3>> &faces,
                       unsigned int bucketSize, int maxDepth) {
    if (n == nullptr || maxDepth == 0 || n->indices.size() < bucketSize)
      return;
    if (n->isLeaf) {
      return; // no splits for leafs
    }

    // the comments are all for the case splitAxis = x so I can explain it to
    // myself
    Point p1 = bboxMin;
    p1[n->splitAxis] =
        getPoints()[n->pointIndex][n->splitAxis]; // (new_x, y_min, z_min)
    Point p2 = bboxMax;
    p2[n->splitAxis] =
        getPoints()[n->pointIndex][n->splitAxis]; // (new_x, y_max, z_max)
    Point p3 = p1; // just saves a line of code for setting x | Since x is
                   // always the same value
    p3[(n->splitAxis + 1) % 3] =
        bboxMax[(n->splitAxis + 1) % 3]; // (new_x, y_max, z_min)
    Point p4 = p1;                       // same as for p3
    p4[(n->splitAxis + 2) % 3] =
        bboxMax[(n->splitAxis + 2) % 3]; // (new_x, y_min, z_max)

    int base = (int)vertices.size();

    vertices.emplace_back(p1);
    vertices.emplace_back(p2);
    vertices.emplace_back(p3);
    vertices.emplace_back(p4);

    // each plane consists of 2 triangles
    // t1 = p1, p3, p2 | t2 = p1, p2, p4
    faces.push_back({base, base + 1, base + 2});
    faces.push_back({base, base + 1, base + 3});

    // "left" side split
    // change max value since for a left side split the new split plane is on
    // the right
    visualiseBBoxes(n->left.get(), bboxMin, p2, vertices, faces, bucketSize,
                    maxDepth - 1);
    // same logic as for leftSide just switched
    visualiseBBoxes(n->right.get(), p1, bboxMax, vertices, faces, bucketSize,
                    maxDepth - 1);
  }
  void getBounds(Point &bboxMin, Point &bboxMax) const {
    bboxMin = getPoints()[0];
    bboxMax = getPoints()[0];
    for (const Point &p : getPoints()) {
      for (int i = 0; i < 3; i++) {
        if (p[i] < bboxMin[i])
          bboxMin[i] = p[i];
        if (p[i] > bboxMax[i])
          bboxMax[i] = p[i];
      }
    }
  }
};

float wendlandWeight(float d, float r) {
  if (d >= r)
    return 0.0f;
  float t = d / r;
  float term1 = 1.0f - t;
  float term2 = term1 * term1; // (1-t)^2
  term2 *= term2;              // (1-t)^4
  return term2 * (4.0f * t + 1.0f);
}

struct WLSResult {
  Point pt;
  Normal normal;
};

WLSResult approximateWLS(const KDTree &tree, float x, float y, float r) {
  Point queryP = {x, y, 0.0f};
  std::vector<std::size_t> neighbors = tree.collectInRadius2D(queryP, r);
  if (neighbors.empty()) {
    return {{x, y, 0.0f}, {0.0f, 0.0f, 1.0f}};
  }

  int numPoints = neighbors.size();
  if (numPoints < 6) {
    float sumZ = 0.0f;
    float sumW = 0.0f;
    for (size_t idx : neighbors) {
      Point pt = tree.getPoints()[idx];
      float dx = queryP[0] - pt[0];
      float dy = queryP[1] - pt[1];
      float d = std::sqrt(dx * dx + dy * dy);
      float w = wendlandWeight(d, r);
      sumZ += w * pt[2];
      sumW += w;
    }
    return {{x, y, sumW > 0 ? sumZ / sumW : 0.0f}, {0.0f, 0.0f, 1.0f}};
  }

  Eigen::MatrixXf A(numPoints, 6);
  Eigen::VectorXf B(numPoints);
  Eigen::MatrixXf W = Eigen::MatrixXf::Zero(numPoints, numPoints);

  for (int i = 0; i < numPoints; ++i) {
    Point pt = tree.getPoints()[neighbors[i]];
    float u = pt[0] - x;
    float v = pt[1] - y;

    A(i, 0) = 1.0f;
    A(i, 1) = u;
    A(i, 2) = v;
    A(i, 3) = u * u;
    A(i, 4) = u * v;
    A(i, 5) = v * v;

    B(i) = pt[2];

    float dx = queryP[0] - pt[0];
    float dy = queryP[1] - pt[1];
    float d = std::sqrt(dx * dx + dy * dy);
    W(i, i) = wendlandWeight(d, r);
  }

  Eigen::MatrixXf Aw = A;
  Eigen::VectorXf Bw = B;
  for (int i = 0; i < numPoints; ++i) {
    float w_sqrt = std::sqrt(W(i, i));
    Aw.row(i) *= w_sqrt;
    Bw(i) *= w_sqrt;
  }

  Eigen::VectorXf c = Aw.colPivHouseholderQr().solve(Bw);

  if (c.hasNaN()) {
    float sumZ = 0.0f;
    float sumW = 0.0f;
    for (int i = 0; i < numPoints; ++i) {
      sumZ += W(i, i) * B(i);
      sumW += W(i, i);
    }
    return {{x, y, sumW > 0 ? sumZ / sumW : 0.0f}, {0.0f, 0.0f, 1.0f}};
  }

  Normal nrm = {-c(1), -c(2), 1.0f};
  float len = std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
  if (len > 1e-6f) {
    nrm[0] /= len;
    nrm[1] /= len;
    nrm[2] /= len;
  }

  return {{x, y, c(0)}, nrm};
}

Point deCasteljau1D(std::vector<Point> pts, float t) {
  int n = pts.size() - 1;
  // n is degree of the Bezier curve
  if (n < 0)
    return {0, 0, 0};
  // repeat interpolation step from degree 1 to n
  for (int r = 1; r <= n; ++r) {
    // loop through the control points
    for (int i = 0; i <= n - r; ++i) {
      // lin interpolation formula
      for (int k = 0; k < 3; ++k) {
        pts[i][k] = (1.0f - t) * pts[i][k] + t * pts[i + 1][k];
      }
    }
  }
  return pts[0]; // exact point for bezier
}

Point deCasteljauDeriv1D(const std::vector<Point> &pts, float t) {
  int deg = pts.size() - 1;
  if (deg <= 0)
    return {0, 0, 0};
  std::vector<Point> diffs(deg);
  for (int i = 0; i < deg; ++i) {
    for (int k = 0; k < 3; ++k) {
      diffs[i][k] = deg * (pts[i + 1][k] - pts[i][k]);
    }
  }
  return deCasteljau1D(diffs, t);
}

Point evaluateBezier(const std::vector<Point> &controlPoints, int m, int n,
                     float u, float v) {
  std::vector<Point> rowResults(n);
  for (int j = 0; j < n; ++j) {
    std::vector<Point> rowPts(m);
    for (int i = 0; i < m; ++i) {
      rowPts[i] = controlPoints[j * m + i];
    }
    rowResults[j] = deCasteljau1D(rowPts, u);
  }
  return deCasteljau1D(rowResults, v);
}

Normal evaluateBezierNormal(const std::vector<Point> &controlPoints, int m,
                            int n, float u, float v) {
  std::vector<Point> du_rowResults(n);
  std::vector<Point> dv_colResults(m);

  for (int j = 0; j < n; ++j) {
    std::vector<Point> rowPts(m);
    for (int i = 0; i < m; ++i) {
      rowPts[i] = controlPoints[j * m + i];
    }
    du_rowResults[j] = deCasteljauDeriv1D(rowPts, u);
  }
  Point du = deCasteljau1D(du_rowResults, v);

  for (int i = 0; i < m; ++i) {
    std::vector<Point> colPts(n);
    for (int j = 0; j < n; ++j) {
      colPts[j] = controlPoints[j * m + i];
    }
    dv_colResults[i] = deCasteljauDeriv1D(colPts, v);
  }
  Point dv = deCasteljau1D(dv_colResults, u);

  Normal normal;
  normal[0] = du[1] * dv[2] - du[2] * dv[1];
  normal[1] = du[2] * dv[0] - du[0] * dv[2];
  normal[2] = du[0] * dv[1] - du[1] * dv[0];

  float len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                        normal[2] * normal[2]);
  if (len > 1e-6f) {
    normal[0] /= len;
    normal[1] /= len;
    normal[2] /= len;
  }
  return normal;
}

void TestRuntime() {

  std::default_random_engine engine(std::random_device{}());
  std::uniform_real_distribution<float> distribution(-1.f, 1.f);

  std::ofstream Runtime(
      "/Users/scherwing/Uni/Berlin/cg2/CG-TUB/ex_01/skeleton/runtime.txt");
  if (!Runtime.is_open()) {
    polyscope::warning("Failed to open runtime.txt");
    return;
  }
  for (unsigned int n = 1000; n <= 10000000; n = n * 10) {
    std::vector<Point> points;
    for (unsigned int i = 0; i < n; ++i) {
      points.emplace_back(Point{distribution(engine), distribution(engine),
                                distribution(engine)});
    }
    // create Tree
    std::unique_ptr<KDTree> tree = std::make_unique<KDTree>(points);
    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    tree->collectInRadius(points[n / 2], 0.3);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Runtime << "Time for " << n
            << " using KDTree implementation of collectInRadius: "
            << duration.count() << std::endl;
    // start timer
    start = std::chrono::high_resolution_clock::now();
    tree->collectKNearest(points[n / 2], 20);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Runtime << "Time for " << n
            << " Points using KDTree implementation of collectKNearest: "
            << duration.count() << std::endl;

    // create Tree
    std::unique_ptr<SpatialDataStructure> ds =
        std::make_unique<SpatialDataStructure>(points);
    // start timer
    start = std::chrono::high_resolution_clock::now();
    ds->collectInRadius(points[n / 2], 0.3);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Runtime << "Time for " << n
            << " Points using Brute Force implementation of collectInRadius: "
            << duration.count() << std::endl;
    // start timer
    start = std::chrono::high_resolution_clock::now();
    ds->collectKNearest(points[n / 2], 20);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Runtime << "Time for " << n
            << " Points using Brute Force implementation of collectKNearest: "
            << duration.count() << std::endl;
  }
  Runtime.flush();
  Runtime.close();
}
// Application variables
polyscope::PointCloud *pc = nullptr;
std::unique_ptr<KDTree> sds;
std::vector<Point> FilePoints;
std::vector<Normal> FileNormals;

/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value
*/
Point VertexInterp(int isolevel, Point p1, Point p2, double valp1, double valp2)
{
   double mu;
   Point p;

   if (std::abs(isolevel-valp1) < 0.00001)
      return(p1);
   if (std::abs(isolevel-valp2) < 0.00001)
      return(p2);
   if (std::abs(valp1-valp2) < 0.00001)
      return(p1);
   mu = (isolevel - valp1) / (valp2 - valp1);
   p[0] = p1[0] + mu * (p2[0] - p1[0]);
   p[1] = p1[1] + mu * (p2[1] - p1[1]);
   p[2] = p1[2] + mu * (p2[2] - p1[2]);

   return(p);
}
int Polygonise(std::array<int, 8> corners, double isolevel, Triangle *triangles, std::vector<float> gridValues, std::vector<Point> gridPoints){
  int i,ntriang;
  int cubeindex;
  Point vertlist[12];

  cubeindex = 0;

  if (gridValues[corners[0]] < isolevel) cubeindex |= 1;
  if (gridValues[corners[1]] < isolevel) cubeindex |= 2;
  if (gridValues[corners[2]] < isolevel) cubeindex |= 4;
  if (gridValues[corners[3]] < isolevel) cubeindex |= 8;
  if (gridValues[corners[4]] < isolevel) cubeindex |= 16;
  if (gridValues[corners[5]] < isolevel) cubeindex |= 32;
  if (gridValues[corners[6]] < isolevel) cubeindex |= 64;
  if (gridValues[corners[7]] < isolevel) cubeindex |= 128;

  if (edgeTable[cubeindex] == 0)
      return(0);

  /* Find the vertices where the surface intersects the cube */
  if (edgeTable[cubeindex] & 1)
    vertlist[0] = VertexInterp(isolevel,gridPoints[corners[0]],gridPoints[corners[1]],gridValues[corners[0]],gridValues[corners[1]]);
  if (edgeTable[cubeindex] & 2)
      vertlist[1] = VertexInterp(isolevel,gridPoints[corners[1]],gridPoints[corners[2]],gridValues[corners[1]],gridValues[corners[2]]);
  if (edgeTable[cubeindex] & 4)
      vertlist[2] = VertexInterp(isolevel,gridPoints[corners[2]],gridPoints[corners[3]],gridValues[corners[2]],gridValues[corners[3]]);
  if (edgeTable[cubeindex] & 8)
      vertlist[3] = VertexInterp(isolevel,gridPoints[corners[3]],gridPoints[corners[0]],gridValues[corners[3]],gridValues[corners[0]]);
  if (edgeTable[cubeindex] & 16)
      vertlist[4] = VertexInterp(isolevel,gridPoints[corners[4]],gridPoints[corners[5]],gridValues[corners[4]],gridValues[corners[5]]);
  if (edgeTable[cubeindex] & 32)
      vertlist[5] = VertexInterp(isolevel,gridPoints[corners[5]],gridPoints[corners[6]],gridValues[corners[5]],gridValues[corners[6]]);
  if (edgeTable[cubeindex] & 64)
      vertlist[6] = VertexInterp(isolevel,gridPoints[corners[6]],gridPoints[corners[7]],gridValues[corners[6]],gridValues[corners[7]]);
  if (edgeTable[cubeindex] & 128)
      vertlist[7] = VertexInterp(isolevel,gridPoints[corners[7]],gridPoints[corners[4]],gridValues[corners[7]],gridValues[corners[4]]);
  if (edgeTable[cubeindex] & 256)
      vertlist[8] = VertexInterp(isolevel,gridPoints[corners[0]],gridPoints[corners[4]],gridValues[corners[0]],gridValues[corners[4]]);
  if (edgeTable[cubeindex] & 512)
      vertlist[9] = VertexInterp(isolevel,gridPoints[corners[1]],gridPoints[corners[5]],gridValues[corners[1]],gridValues[corners[5]]);
  if (edgeTable[cubeindex] & 1024)
      vertlist[10] = VertexInterp(isolevel,gridPoints[corners[2]],gridPoints[corners[6]],gridValues[corners[2]],gridValues[corners[6]]);
  if (edgeTable[cubeindex] & 2048)
      vertlist[11] = VertexInterp(isolevel,gridPoints[corners[3]],gridPoints[corners[7]],gridValues[corners[3]],gridValues[corners[7]]);

  /* Create the triangle */
   ntriang = 0;
   for (i=0;triTable[cubeindex][i]!=-1;i+=3) {
      triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i  ]];
      triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i+1]];
      triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i+2]];
      ntriang++;
   }

   return(ntriang);
}
float evaluateImplicit(const Point& p, const KDTree& constraintSDS, float wlsRadius, const std::vector<Point> constraintPoints, const std::vector<float> constraintValues) {
      std::vector<std::size_t> neighbors = constraintSDS.collectInRadius(p, wlsRadius);
          
      float sumW = 0.0f;
      float sumWF = 0.0f;
          
      for (std::size_t idx : neighbors) {
          float dist = EuclideanDistance::measure(p, constraintPoints[idx]);
          float w = wendlandWeight(dist, wlsRadius);
          sumWF += w * constraintValues[idx];
          sumW += w;
      }
          
      return sumW > 0 ? sumWF / sumW : 0.0f;
}
void callback() {

  if (ImGui::Button("Load Off")) {
    auto paths =
        pfd::open_file("Load Off", "",
                       std::vector<std::string>{"point data (*.off)", "*.off"},
                       pfd::opt::none)
            .result();
    if (!paths.empty()) {
      std::filesystem::path path(paths[0]);

      if (path.extension() == ".off") {
        // Read the point cloud
        readOff(path.string(), FilePoints);

        // Create the polyscope geometry
        pc = polyscope::registerPointCloud("Points", FilePoints);

        // Build spatial data structure
        sds = std::make_unique<KDTree>(FilePoints);
      }
    }
  }

  static float radius = 1.0;
  static int k = 5;
  auto selection = polyscope::getSelection();
  static int searchMode = 0;
  const char *modes[] = {"Collect In Radius", "Collect K Nearest", "None"};
  ImGui::Combo("Search Mode", &searchMode, modes, 3);

  if (searchMode == 0) {
    ImGui::SliderFloat("Radius", &radius, 0.001f, 1000.0f);
  } else {
    ImGui::SliderInt("K", &k, 1, 1000);
  }

  if (selection.isHit && pc != nullptr && searchMode != 2) {
    size_t index = selection.localIndex;
    Point selectedPoint = sds->getPoints()[index];
    std::vector<std::size_t> highlight;
    if (searchMode == 0) {
      highlight = sds->collectInRadius(selectedPoint, radius);

    } else if (searchMode == 1) {
      highlight = sds->collectKNearest(selectedPoint, k);
    } else {
      highlight = {};
    }
    std::vector<float> colors(sds->getPoints().size(),
                              0.0f); // all points unselected
    // set selected points to 1
    for (std::size_t idx : highlight) {
      colors[idx] = 1.0f;
    }
    pc->addScalarQuantity("selection", colors);
    pc->addScalarQuantity("selection", colors)->setEnabled(true);
  }

  // buttons and vars for kdTree
  static int treeDepth = 5;
  static int bucketSize = 3;

  if (ImGui::Button("Test Runtime")) {
    TestRuntime();
  }
  if (ImGui::Button("Visualise KD-Tree")) {
    std::vector<Point> vertices;
    std::vector<std::array<int, 3>> faces;
    Point bboxMin, bboxMax;
    sds->getBounds(bboxMin, bboxMax);
    sds->visualiseBBoxes(bboxMin, bboxMax, vertices, faces, bucketSize,
                         treeDepth);
    polyscope::registerSurfaceMesh("KDTree Planes", vertices, faces);
  }
  bool changed = ImGui::SliderInt("Tree Depth", &treeDepth, 1, 200);
  changed |= ImGui::SliderInt("Bucket Size", &bucketSize, 1, 200);
  if (changed && sds != nullptr) {
    std::vector<Point> vertices;
    std::vector<std::array<int, 3>> faces;
    Point bboxMin, bboxMax;
    sds->getBounds(bboxMin, bboxMax);
    sds->visualiseBBoxes(bboxMin, bboxMax, vertices, faces, bucketSize,
                         treeDepth);
    polyscope::registerSurfaceMesh("KDTree Planes", vertices, faces);
  }

  //------------------- ex_02------------------

  ImGui::Separator();
  ImGui::Text("Exercise 2: Surfaces");
  static int gridN = 10;
  static int gridM = 10;
  static int kMesh = 5;

  static float wlsRadius = 0.5f;

  ImGui::SliderInt("m", &gridM, 2, 50);
  ImGui::SliderInt("n", &gridN, 2, 50);
  ImGui::SliderInt("k", &kMesh, 1, 20);
  ImGui::SliderFloat("WLS Radius", &wlsRadius, 0.01f, 1000.0f);

  if (ImGui::Button("Generate Bezier Tensor Product Surface")) {
    if (sds != nullptr) {
      Point bboxMin, bboxMax;
      sds->getBounds(bboxMin, bboxMax);

      std::vector<Point> controlPoints(gridM * gridN);
      std::vector<Point> flatControlPoints(gridM * gridN);

      std::vector<std::future<void>> futuresCtrl;
      int numThreads = std::thread::hardware_concurrency();
      if (numThreads == 0)
        numThreads = 4;
      int chunkCtrl = gridN / numThreads;
      if (chunkCtrl == 0)
        chunkCtrl = 1;

      for (int t = 0; t < numThreads; ++t) {
        int startJ = t * chunkCtrl;
        int endJ = (t == numThreads - 1) ? gridN : startJ + chunkCtrl;
        if (startJ >= gridN)
          break;

        futuresCtrl.push_back(std::async(
            std::launch::async, [startJ, endJ, bboxMin, bboxMax, &controlPoints,
                                 &flatControlPoints]() {
              for (int j = startJ; j < endJ; ++j) {
                float v =
                    (gridN == 1) ? 0.5f : static_cast<float>(j) / (gridN - 1);
                float y = bboxMin[1] + v * (bboxMax[1] - bboxMin[1]);
                for (int i = 0; i < gridM; ++i) {
                  float u =
                      (gridM == 1) ? 0.5f : static_cast<float>(i) / (gridM - 1);
                  float x = bboxMin[0] + u * (bboxMax[0] - bboxMin[0]);
                  WLSResult res = approximateWLS(*sds, x, y, wlsRadius);
                  controlPoints[j * gridM + i] = res.pt;
                  flatControlPoints[j * gridM + i] = {x, y, bboxMin[2]};
                }
              }
            }));
      }
      for (auto &f : futuresCtrl) {
        f.get();
      }

      int numU = kMesh * gridM;
      int numV = kMesh * gridN;

      std::vector<Point> surfVertices(numU * numV);
      std::vector<Normal> surfNormals(numU * numV);

      // numV is resolution of v
      // find out where we are in the grid
      for (int j = 0; j < numV; ++j) {
        float v = (numV == 1) ? 0.5f : static_cast<float>(j) / (numV - 1);

        for (int i = 0; i < numU; ++i) {
          float u = (numU == 1) ? 0.5f : static_cast<float>(i) / (numU - 1);

          surfVertices[j * numU + i] =
              evaluateBezier(controlPoints, gridM, gridN, u, v);
          surfNormals[j * numU + i] =
              evaluateBezierNormal(controlPoints, gridM, gridN, u, v);
        }
      }

      std::vector<std::vector<size_t>> surfFaces;
      for (int j = 0; j < numV - 1; ++j) {
        for (int i = 0; i < numU - 1; ++i) {
          size_t idx0 = j * numU + i;
          size_t idx1 = j * numU + (i + 1);
          size_t idx2 = (j + 1) * numU + (i + 1);
          size_t idx3 = (j + 1) * numU + i;
          surfFaces.push_back({idx0, idx1, idx2, idx3});
        }
      }

      auto surfMesh = polyscope::registerSurfaceMesh("Bezier Surface",
                                                     surfVertices, surfFaces);
      surfMesh->addVertexVectorQuantity("normals", surfNormals);

      // edges for control points
      std::vector<std::array<size_t, 2>> ctrlEdges;
      // for every controllpoint
      // gridN -> x-axe points (rows)
      // gridM -> y-axe points (columns)
      for (int j = 0; j < gridN; ++j) {
        for (int i = 0; i < gridM; ++i) {
          size_t idx = j * gridM + i;
          if (i < gridM - 1)
            ctrlEdges.push_back({idx, idx + 1});
          if (j < gridN - 1)
            ctrlEdges.push_back({idx, idx + gridM});
        }
      }

      auto ctrlNet = polyscope::registerCurveNetwork("Control Grid (3D)",
                                                     controlPoints, ctrlEdges);
      ctrlNet->setRadius(0.002);

      auto flatNet = polyscope::registerCurveNetwork(
          "Control Grid (2D)", flatControlPoints, ctrlEdges);
      flatNet->setRadius(0.002);
      flatNet->setEnabled(false); // Hidden by default

      auto ctrlPoints =
          polyscope::registerPointCloud("Control Points", controlPoints);
      ctrlPoints->setPointRadius(0.01);
    }
  }

  if (ImGui::Button("Generate MLS Surface")) {
    if (sds != nullptr) {
      Point bboxMin, bboxMax;
      sds->getBounds(bboxMin, bboxMax);

      int numU = kMesh * gridM;
      int numV = kMesh * gridN;
      std::vector<Point> mlsVertices(numU * numV);
      std::vector<Normal> mlsNormals(numU * numV);

      std::vector<std::future<void>> futures;
      int numThreads = std::thread::hardware_concurrency();
      if (numThreads == 0)
        numThreads = 4;
      int chunk = numV / numThreads;
      if (chunk == 0)
        chunk = 1;

      for (int t = 0; t < numThreads; ++t) {
        int startJ = t * chunk;
        int endJ = (t == numThreads - 1) ? numV : startJ + chunk;
        if (startJ >= numV)
          break;

        futures.push_back(std::async(std::launch::async, [startJ, endJ, numU,
                                                          numV, bboxMin,
                                                          bboxMax, &mlsVertices,
                                                          &mlsNormals]() {
          for (int j = startJ; j < endJ; ++j) {

            float v = (numV == 1) ? 0.5f : static_cast<float>(j) / (numV - 1);
            float y = bboxMin[1] + v * (bboxMax[1] - bboxMin[1]);
            for (int i = 0; i < numU; ++i) {
              float u = (numU == 1) ? 0.5f : static_cast<float>(i) / (numU - 1);
              float x = bboxMin[0] + u * (bboxMax[0] - bboxMin[0]);
              WLSResult res = approximateWLS(*sds, x, y, wlsRadius);
              mlsVertices[j * numU + i] = res.pt;
              mlsNormals[j * numU + i] = res.normal;
            }
          }
        }));
      }
      for (auto &f : futures) {
        f.get();
      }

      std::vector<std::vector<size_t>> mlsFaces;
      for (int j = 0; j < numV - 1; ++j) {
        for (int i = 0; i < numU - 1; ++i) {
          size_t idx0 = j * numU + i;
          size_t idx1 = j * numU + (i + 1);
          size_t idx2 = (j + 1) * numU + (i + 1);
          size_t idx3 = (j + 1) * numU + i;
          mlsFaces.push_back({idx0, idx1, idx2, idx3});
        }
      }

      auto mlsMesh =
          polyscope::registerSurfaceMesh("MLS Surface", mlsVertices, mlsFaces);
      mlsMesh->addVertexVectorQuantity("normals", mlsNormals);
    }
  }

  //------------------- ex_03------------------

  ImGui::Separator();
  ImGui::Text("Exercise 3: Approximation");

  if (ImGui::Button("Load NOFF")) {
    auto paths =
        pfd::open_file("Load Off", "",
                       std::vector<std::string>{"point data (*.off)", "*.off"},
                       pfd::opt::none)
            .result();
    if (!paths.empty()) {
      std::filesystem::path path(paths[0]);

      if (path.extension() == ".off") {
        // Read the point cloud
        readOff(path.string(), FilePoints, FileNormals);

        // Create the polyscope geometry
        pc = polyscope::registerPointCloud("Points", FilePoints);
        if (!FileNormals.empty()) {
          pc->addVectorQuantity("Normals", FileNormals);
        }
        // Build spatial data structure
        sds = std::make_unique<KDTree>(FilePoints);
      }
    }
  }

  std::vector<float> constraintValues;
  std::vector<Point> constraintPoints;
  std::vector<Point> gridPoints;
  std::vector<Point> meshVertices;
  std::unique_ptr<KDTree> constraintSDS = nullptr;
  std::vector<std::array<size_t, 3>> meshFaces;

  static int numGridPoints = 10;
  ImGui::SliderInt("Grid Points", &numGridPoints, 2, 100);

  if (ImGui::Button("Generate Constraints and Grid")) {
    constraintPoints.clear();
    constraintValues.clear();
    gridPoints.clear();
    meshVertices.clear();
    meshFaces.clear();
    if (sds != nullptr) {
      Point bboxMin;
      Point bboxMax;
      sds->getBounds(bboxMin, bboxMax);

      float alpha = 0.01 * EuclideanDistance().measure(bboxMin, bboxMax);
      std::ofstream log("/Users/scherwing/Uni/Berlin/cg2/CG-TUB/ex_01/skeleton/debug.txt");
      log << alpha << std::endl;
      for (unsigned int i = 0; i < FilePoints.size(); i++) {
        float PosAlpha = alpha;
        Point pPos;
        int posIter = 0;
        while(true) {
          pPos = {
            FilePoints[i][0] + FileNormals[i][0] * PosAlpha,
            FilePoints[i][1] + FileNormals[i][1] * PosAlpha,
            FilePoints[i][2] + FileNormals[i][2] * PosAlpha
          };
          if (sds->collectKNearest(pPos, 1)[0] == i) {
            break;
          }
          PosAlpha /= 2;
          posIter++;
          if (posIter > 50) {
              log << "Point " << i << " pos loop exceeded 50 iterations" << std::endl;
              break;
          }
        }

        float NegAlpha = alpha;
        Point pNeg;
        int negIter = 0;
        while (true) {
          pNeg = {
            FilePoints[i][0] - FileNormals[i][0] * NegAlpha,
            FilePoints[i][1] - FileNormals[i][1] * NegAlpha,
            FilePoints[i][2] - FileNormals[i][2] * NegAlpha
          };
          if (sds->collectKNearest(pNeg, 1)[0] == i) {
            break;
          }
          NegAlpha /= 2;
          negIter++;
          if (negIter > 50) {
              log << "Point " << i << " neg loop exceeded 50 iterations" << std::endl;
              break;
          }
        }
        constraintPoints.emplace_back(FilePoints[i]);
        constraintValues.emplace_back(0.0f);
    
        constraintPoints.emplace_back(pPos);
        constraintValues.emplace_back(PosAlpha);
    
        constraintPoints.emplace_back(pNeg);
        constraintValues.emplace_back(-NegAlpha);
      }
      log.close();
      // generate grid

      // make bbox slightly larger
      float margin = EuclideanDistance().measure(bboxMin, bboxMax) *0.05;
      bboxMin[0] -= margin;
      bboxMin[1] -= margin;
      bboxMin[2] -= margin;
      bboxMax[0] += margin;
      bboxMax[1] += margin;
      bboxMax[2] += margin;

      for (int iz = 0; iz < numGridPoints; iz++) {
        for (int iy = 0; iy < numGridPoints; iy++) {
            for (int ix = 0; ix < numGridPoints; ix++) {
                float x = bboxMin[0] + (float)ix / (numGridPoints-1) * (bboxMax[0] - bboxMin[0]);
                float y = bboxMin[1] + (float)iy / (numGridPoints-1) * (bboxMax[1] - bboxMin[1]);
                float z = bboxMin[2] + (float)iz / (numGridPoints-1) * (bboxMax[2] - bboxMin[2]);
                gridPoints.emplace_back(Point{x, y, z});
            }
        }
      }
      constraintSDS = std::make_unique<KDTree>(constraintPoints);
      std::vector<float> gridValues;

      for (const Point& gridPoint : gridPoints) {
          // find all constraint points within radius
          std::vector<std::size_t> neighbors = constraintSDS->collectInRadius(gridPoint, wlsRadius);
          
          float sumW = 0.0f;
          float sumWF = 0.0f;
          
          for (std::size_t idx : neighbors) {
              float dist = EuclideanDistance::measure(gridPoint, constraintPoints[idx]);
              float w = wendlandWeight(dist, wlsRadius);
              sumWF += w * constraintValues[idx];
              sumW += w;
          }
          
          gridValues.emplace_back(sumW > 0 ? sumWF / sumW : 0.0f);
      }
      auto gridPC = polyscope::registerPointCloud("Grid Points", gridPoints);
      gridPC->addScalarQuantity("implicit function", gridValues)->setEnabled(true);

      // apply marching cubes algorithm
      // returns index of point at pos x,y,z
      auto idx = [&](int x, int y, int z) {
          return z * numGridPoints * numGridPoints + y * numGridPoints + x;
      };

      Triangle triangles[5]; // max 5 triangles per cube

      for (int iz = 0; iz < numGridPoints-1; iz++) {
        for (int iy = 0; iy < numGridPoints-1; iy++) {
            for (int ix = 0; ix < numGridPoints-1; ix++) {
              // calculate corners of cube
              int c0 = idx(ix,   iy,   iz  );
              int c1 = idx(ix+1, iy,   iz  );
              int c2 = idx(ix+1, iy+1, iz  );
              int c3 = idx(ix,   iy+1, iz  );
              int c4 = idx(ix,   iy,   iz+1);
              int c5 = idx(ix+1, iy,   iz+1);
              int c6 = idx(ix+1, iy+1, iz+1);
              int c7 = idx(ix,   iy+1, iz+1);

              std::array<int, 8> corners = {c0, c1, c2, c3, c4, c5, c6, c7};

              int n = Polygonise(corners, 0.0, triangles, gridValues, gridPoints);

              for (int t = 0; t < n; t++) {
                size_t base = meshVertices.size();
                meshVertices.push_back(triangles[t].p[0]);
                meshVertices.push_back(triangles[t].p[1]);
                meshVertices.push_back(triangles[t].p[2]);
                meshFaces.push_back({base, base+1, base+2});
              }
            }
        }
      }
      std::vector<Point> vertexNormals;
      for (Point vertice : meshVertices) {
        float epsilon = 0.001;
        Point p1 = {
          vertice[0] + epsilon,
          vertice[1] + epsilon,
          vertice[2] + epsilon
        };
        Point p2 = {
          vertice[0] - epsilon,
          vertice[1] - epsilon,
          vertice[2] - epsilon
        };
        float dx = evaluateImplicit({vertice[0]+epsilon, vertice[1], vertice[2]}, *constraintSDS, wlsRadius, constraintPoints, constraintValues)
        - evaluateImplicit({vertice[0]-epsilon, vertice[1], vertice[2]}, *constraintSDS, wlsRadius, constraintPoints, constraintValues);

        float dy = evaluateImplicit({vertice[0], vertice[1]+epsilon, vertice[2]}, *constraintSDS, wlsRadius, constraintPoints, constraintValues)
                - evaluateImplicit({vertice[0], vertice[1]-epsilon, vertice[2]}, *constraintSDS, wlsRadius, constraintPoints, constraintValues);

        float dz = evaluateImplicit({vertice[0], vertice[1], vertice[2]+epsilon}, *constraintSDS, wlsRadius, constraintPoints, constraintValues)
                - evaluateImplicit({vertice[0], vertice[1], vertice[2]-epsilon}, *constraintSDS, wlsRadius, constraintPoints, constraintValues);

        Normal normal = {dx, dy, dz};

        float len = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (len > 1e-6f) { normal[0]/=len; normal[1]/=len; normal[2]/=len; }

        vertexNormals.emplace_back(normal);
      }
      auto mesh = polyscope::registerSurfaceMesh("Marching Cubes", meshVertices, meshFaces);
      mesh->addVertexVectorQuantity("normals", vertexNormals);      
    }
  }
}

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Computer Graphics 2 Sample Code.");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help &) {
    std::cout << parser;
    return 0;
  } catch (const args::ParseError &e) {
    std::cerr << e.what() << std::endl;

    std::cerr << parser;
    return 1;
  }

  // Options

  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
  polyscope::options::shadowBlurIters = 6;

  // Initialize polyscope
  polyscope::init();

  // Add a few gui elements
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
