
#include <utility>
#include <vector>
#include <fstream>
#include <colmap/base/database.h>
#include <Eigen/Core>
#include <colmap/base/camera.h>
#include <colmap/estimators/two_view_geometry.h>
#include <Eigen/Geometry>

class Estimator
{
public:
    explicit Estimator(
            const colmap::Database& database,
            const colmap::image_t& image_id1,
            const colmap::image_t& image_id2,
            colmap::FeatureMatches  matches)
            : database_(database), image_id1_(image_id1), image_id2_(image_id2), matches_(std::move(matches)) {}

    void run()
    {
        readImagesAndCameras();
        readKeypointsAndMatches();
        estimateTwoViewGeometry();

    }

    void run_next(){
        readImagesAndCameras();
        readKeypointsAndMatches();
    }

    Eigen::Vector3d getRotationVector()
    {
        if (two_view_geometry_.config != colmap::TwoViewGeometry::ConfigurationType::UNDEFINED) {
            Eigen::Quaterniond q(two_view_geometry_.qvec[0], two_view_geometry_.qvec[1], two_view_geometry_.qvec[2], two_view_geometry_.qvec[3]);
            Eigen::AngleAxisd angle_axis(q);
            return angle_axis.axis() * angle_axis.angle();
        } else {
            // Return a zero vector if the pose estimation failed
            return Eigen::Vector3d::Zero();
        }
    }

    Eigen::Vector3d getTranslationVector()
    {
        if (two_view_geometry_.config != colmap::TwoViewGeometry::ConfigurationType::UNDEFINED) {
            return two_view_geometry_.tvec;
        } else {
            // Return a zero vector if the pose estimation failed
            return Eigen::Vector3d::Zero();
        }
    }

    std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> getKeyPoints(){
        return std::make_pair(points1_,points2_);
    }

private:
    const colmap::Database& database_;
    colmap::image_t image_id1_, image_id2_;
    colmap::Image image1_, image2_;
    colmap::Camera camera1_, camera2_;
    colmap::FeatureKeypoints keypoints1_, keypoints2_;
    colmap::FeatureMatches matches_;
    std::vector<Eigen::Vector2d> points1_, points2_;
    colmap::TwoViewGeometry two_view_geometry_;

    void readImagesAndCameras()
    {
        image1_ = database_.ReadImage(image_id1_);
        image2_ = database_.ReadImage(image_id2_);

        camera1_ = database_.ReadCamera(image1_.CameraId());
        camera2_ = database_.ReadCamera(image2_.CameraId());
    }

    void readKeypointsAndMatches()
    {
        keypoints1_ = database_.ReadKeypoints(image_id1_);
        keypoints2_ = database_.ReadKeypoints(image_id2_);
        matches_ = database_.ReadMatches(image_id1_, image_id2_);

        for (auto& match : matches_) {
            points1_.emplace_back(keypoints1_[match.point2D_idx1].x, keypoints1_[match.point2D_idx1].y);
            points2_.emplace_back(keypoints2_[match.point2D_idx2].x, keypoints2_[match.point2D_idx2].y);
        }
    }

    void estimateTwoViewGeometry()
    {
        std::cout<<"two view geometry might take long time"<<std::endl;
        colmap::TwoViewGeometry::Options options;
        options.ransac_options.max_error = 1.0;
        options.ransac_options.min_inlier_ratio = 0.1;

        two_view_geometry_.Estimate(camera1_, points1_, camera2_, points2_, matches_, options);
        two_view_geometry_.EstimateRelativePose(camera1_, points1_, camera2_, points2_);
    }

};

