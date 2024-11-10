#include <iostream>
#include <vector>

#include <colmap/base/database.h>
#include <fstream>

class DatabaseReader {
public:
    explicit DatabaseReader(const std::string& database_path)
            : database_(database_path) {}

    std::vector<colmap::Camera> ReadAllCameras() {
        return database_.ReadAllCameras();
    }

    Eigen::Matrix3d GetCalibrationMatrix(const colmap::Camera& camera) {
        return camera.CalibrationMatrix();
    }

    colmap::Image ReadImage(colmap::image_t image_id){
        return database_.ReadImage(image_id);
    }

    colmap::Camera ReadCamera(colmap::camera_t camera_id){
        return database_.ReadCamera(camera_id);
    }


    std::vector<std::pair<colmap::image_pair_t, colmap::FeatureMatches>> ReadAllMatches() {
        return database_.ReadAllMatches();
    }

    colmap::Database& GetDatabase() { return database_; }

    std::vector<colmap::Image> ReadAllImage(){
        return database_.ReadAllImages();
    }

    std::pair<colmap::image_t, colmap::image_t> GetImagePair(
            const colmap::image_pair_t image_pair_id) {
        colmap::image_t image_id1, image_id2;
        colmap::Database::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);
        return std::make_pair(image_id1, image_id2);
    }

private:
    colmap::Database database_;
};
