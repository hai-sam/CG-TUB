#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "args/args.hxx"
#include "polyscope/combining_hash_functions.h"
#include "polyscope/curve_network.h"
#include "polyscope/file_helpers.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "portable-file-dialogs.h"

using Point = std::array<float, 3>;
using Normal = std::array<float, 3>;

void readOff(const std::string &filename, std::vector<Point> &points,
             std::vector<Normal> &normals) {
  points.clear();
  normals.clear();

  std::string line;
  std::ifstream file(filename);

  std::getline(file, line);
  std::getline(file, line);

  std::istringstream meta(
      line); // read vertices and faces idk if its needed | edges isnt used
  int Nvertices, Nfaces, Nedges;
  meta >> Nvertices >> Nfaces >> Nedges;

  for (int i = 0; i < Nvertices; i++) {
    std::getline(file, line);
    std::istringstream pointLine(line);
    float x, y, z;
    pointLine >> x >> y >> z;

    points.emplace_back(Point{x, y, z});
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
  // loop from degree 1 up to n
  for (int r = 1; r <= n; ++r) {
    // loop through the control points
    for (int i = 0; i <= n - r; ++i) {
      for (int k = 0; k < 3; ++k) {
        pts[i][k] = (1.0f - t) * pts[i][k] + t * pts[i + 1][k];
      }
    }
  }
  return pts[0];
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
        std::vector<Point> points;
        readOff(path.string(), points);

        // Create the polyscope geometry
        pc = polyscope::registerPointCloud("Points", points);

        // Build spatial data structure
        sds = std::make_unique<KDTree>(points);
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
    ImGui::SliderFloat("Radius", &radius, 0.001f, 1.0f);
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
  ImGui::SliderFloat("WLS Radius", &wlsRadius, 0.01f, 5.0f);

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

      std::vector<std::array<size_t, 2>> ctrlEdges;
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
