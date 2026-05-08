#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "args/args.hxx"
#include "polyscope/combining_hash_functions.h"
#include "polyscope/file_helpers.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "portable-file-dialogs.h"

using Point = std::array<float, 3>;
using Normal = std::array<float, 3>;

void readOff(const std::string& filename, std::vector<Point>& points, std::vector<Normal>& normals) {
    points.clear();
    normals.clear();

    std::string line;
    std::ifstream file(filename);

    std::getline(file,line);
    std::getline(file,line);

    std::istringstream meta(line); // read vertices and faces idk if its needed | edges isnt used
    int Nvertices,Nfaces, Nedges;
    meta >> Nvertices >> Nfaces >> Nedges;

    for (int i = 0; i < Nvertices; i++){
        std::getline(file, line);
        std::istringstream pointLine(line);
        float x, y, z;
        pointLine >> x >> y >> z;

        points.emplace_back(Point{
            x,
            y,
            z
        });
    }
}

void readOff(const std::string& filename, std::vector<Point>& points) {
    std::vector<Normal> dummy_normals;
    readOff(filename, points, dummy_normals);  // reuse the full version
}

struct EuclideanDistance {
    static float measure(Point const& p1, Point const& p2) {
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
    SpatialDataStructure(std::vector<Point> const& points) : m_points(points) {}

    virtual ~SpatialDataStructure() = default;

    std::vector<Point> const& getPoints() const { return m_points; }

    virtual std::vector<std::size_t> collectInRadius(Point const& p, float radius) const {
        std::vector<std::size_t> result;

        // Dummy brute-force implementation
        // Implemented in kdTree subclass
        for (std::size_t i = 0; i < m_points.size(); ++i) {
            float distance = EuclideanDistance::measure(p, m_points[i]);
            if (distance <= radius) result.push_back(i);
        }

        return result;
    }

    virtual std::vector<std::size_t> collectKNearest(Point const& p, unsigned int k) const {

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

struct Node { // to represent the Nodes in the kd-tree (if both child nodes are null -> leaf node)
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    int pointIndex;
    bool isLeaf;
    int splitAxis;
};

class KDTree : public SpatialDataStructure {
    private:
    std::unique_ptr<Node> root;
    public:
    KDTree(std::vector<Point> const& points) : SpatialDataStructure(points) {
        std::vector<std::size_t> indices(points.size());
        for (std::size_t i = 0; i < points.size(); i++) indices[i] = i;
        root = BuildTree(indices, 0);
    }
    std::unique_ptr<Node> BuildTree(std::vector<std::size_t> indices, int axis){
        if (indices.size() == 0) return nullptr;
        Node n;
        if (indices.size() <= 1) {
            n.isLeaf = true;
            n.pointIndex = indices[0];
            return std::make_unique<Node>(std::move(n));
        }
        n.splitAxis = axis;
        std::vector<std::size_t> left;
        std::vector<std::size_t> right;

        for (unsigned int i = 0; i < indices.size(); i++){
            if (i == indices.size()/2) {
                n.pointIndex = indices[indices.size()/2];
            }
            else if (getPoints()[indices[i]][axis] < getPoints()[indices[indices.size()/2]][axis]) {
                left.emplace_back(indices[i]);
            } else {
                right.emplace_back(indices[i]);
            }
        }
        n.isLeaf = false;
        if (left.size()) {
            n.left = BuildTree( left, (axis + 1) % 3);
        }
        if (right.size()) {
            n.right = BuildTree( right, (axis + 1) % 3);
        }
        return std::make_unique<Node>(std::move(n));
    }
    std::vector<std::size_t> collectKNearest(Point const& p, unsigned int k) const {
        std::vector<std::size_t> indices;
        std::priority_queue<std::pair<float, std::size_t>> pq;
        SearchTreeK(root.get(), p, pq, k);
        while (!pq.empty()) {
            indices.push_back(pq.top().second);
            pq.pop();
        }
    return indices;
    }
    std::vector<std::size_t> collectInRadius(Point const& p, float radius) const {
        std::vector<std::size_t> indices;

        return indices;
    }
    void SearchTreeRadius(Node* n, Point const& p, std::vector<size_t>& result, float radius) {
        if (n == nullptr) return;
        float dist = EuclideanDistance::measure(p, getPoints()[n->pointIndex]);
        if (dist < radius) {
            result.push_back(n->pointIndex);
        }
        if (p[n->splitAxis] < getPoints()[n->pointIndex][n->splitAxis]) {
            SearchTreeRadius(n->left.get(), p, result, radius);
            float planeDistance = abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
            if (planeDistance < radius) {
                SearchTreeRadius(n->right.get(), p, result, radius);
            }
        } else {
            SearchTreeRadius(n->right.get(), p, result, radius);
            float planeDistance = abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
            if (planeDistance < radius) {
                SearchTreeRadius(n->left.get(), p, result, radius);
            }
        }
    }
    void SearchTreeK(Node* n, Point const& p, std::priority_queue<std::pair<float, std::size_t>>& result, unsigned int k) const {
        if (n == nullptr) return;
        float dist = EuclideanDistance::measure(p, getPoints()[n->pointIndex]);
        if (result.size() < k || dist < result.top().first) {
            result.push({dist, n->pointIndex});
            if (result.size() > k) result.pop(); // remove farthest
        }
        if (p[n->splitAxis] < getPoints()[n->pointIndex][n->splitAxis]) {
            SearchTreeK(n->left.get(), p, result, k);
            float planeDistance = abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
            if (result.size() < k || planeDistance < result.top().first) {
                SearchTreeK(n->right.get(), p, result, k);
            }
        } else {
            SearchTreeK(n->right.get(), p, result, k);
            float planeDistance = abs(p[n->splitAxis] - getPoints()[n->pointIndex][n->splitAxis]);
            if (result.size() < k || planeDistance < result.top().first) {
                SearchTreeK(n->left.get(), p, result, k);
            }
        }
    }
};

void TestRuntime() {

    std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);

    for (unsigned int n = 100; n <= 10000; n = n * 10){
        std::vector<Point> points;
        for (unsigned int i = 0; i < n; ++i) {
            points.emplace_back(Point{
                distribution(engine),
                distribution(engine),
                distribution(engine)
            });
        }
        // create Tree
        std::unique_ptr<KDTree> tree = std::make_unique<KDTree>(points);
        // start timer
        auto start = std::chrono::high_resolution_clock::now();
        tree->collectInRadius(points[n/2], 0.3);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time for 100 Points using KDTree implementation of collectInRadius: " << duration.count() << std::endl;
        // start timer
        start = std::chrono::high_resolution_clock::now();
        tree->collectKNearest(points[n/2], 20);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time for 100 Points using KDTree implementation of collectKNearest: " << duration.count() << std::endl;

        // create Tree
        std::unique_ptr<SpatialDataStructure> ds = std::make_unique<KDTree>(points);
        // start timer
        start = std::chrono::high_resolution_clock::now();
        ds->collectInRadius(points[n/2], 0.3);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time for 100 Points using Brute Force implementation of collectInRadius: " << duration.count() << std::endl;
        // start timer
        start = std::chrono::high_resolution_clock::now();
        ds->collectKNearest(points[n/2], 20);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time for 100 Points using Brute Force implementation of collectKNearest: " << duration.count() << std::endl;

    }
}

// Application variables
polyscope::PointCloud* pc = nullptr;
std::unique_ptr<SpatialDataStructure> sds;

void callback() {
    if (ImGui::Button("Load Off")) {
        auto paths =
            pfd::open_file("Load Off", "", std::vector<std::string>{"point data (*.off)", "*.off"}, pfd::opt::none)
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


    // TODO: Implement radius search
    // TODO: Implement visualizations
}

int main(int argc, char** argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Computer Graphics 2 Sample Code.");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help&) {
    std::cout << parser;
    return 0;
  } catch (const args::ParseError& e) {
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
