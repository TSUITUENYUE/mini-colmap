
#include <utility>
#include <vector>
#include <fstream>
#include <colmap/base/database.h>
#include <Eigen/Core>
#include <colmap/base/camera.h>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

// triangulation should consider both absolute pose
// TO DO HERE
class Triangulation{
public:
    explicit Triangulation(
            const std::vector<Eigen::Vector2d>& points1,
            const std::vector<Eigen::Vector2d>& points2,
            Eigen::Matrix3d  intrinsics,
            Eigen::Matrix3d  rotation1,
            Eigen::Vector3d   translation1,
            Eigen::Matrix3d  rotation2,
            Eigen::Vector3d   translation2)
            :points1_(points1), points2_(points2), intrinsics_(std::move(intrinsics)), rotation1_(std::move(rotation1)),
            translation1_(std::move(translation1)),rotation2_(std::move(rotation2)),translation2_(std::move(translation2)) {}


    virtual void run(){
        world_coord_ = getWorldCoord();
    }
    std::vector<Eigen::Vector3d> getWorldCoord(){

        // Perform linear triangulation for each points
        std::vector<Eigen::Vector3d> world_coords;
        for (int i = 0; i < points1_.size(); i ++) {
            world_coords.emplace_back(LinearTriangulation(points1_[i], points2_[i], rotation1_, translation1_, rotation2_, translation2_, intrinsics_));
        }
        return world_coords;
    }

    std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>> getPointsPair(){
        return std::make_pair(points2_, world_coord_);
    }



private:
    std::vector<Eigen::Vector2d> points1_;
    std::vector<Eigen::Vector2d> points2_;
    Eigen::Matrix3d intrinsics_;
    Eigen::Matrix3d rotation1_;
    Eigen::Vector3d translation1_;
    Eigen::Matrix3d rotation2_;
    Eigen::Vector3d translation2_;
    std::vector<Eigen::Vector3d> world_coord_;


    static Eigen::Vector3d LinearTriangulation(
                                        const Eigen::Vector2d& point1,
                                        const Eigen::Vector2d& point2,
                                        const Eigen::Matrix3d& rotation1,
                                        const Eigen::Vector3d& translation1,
                                        const Eigen::Matrix3d& rotation2,
                                        const Eigen::Vector3d& translation2,
                                        const Eigen::Matrix3d& intrinsics
                                        ){
        // Convert the points to homogeneous coordinates
        // first8_2d[i] = (K_inv * first8_2d[i].homogeneous()).hnormalized();
        Eigen::Vector3d point1_h = intrinsics.inverse() * (point1.homogeneous());
        Eigen::Vector3d point2_h = intrinsics.inverse() * (point2.homogeneous());
        // std::cout<<point1_h<<std::endl;
        // Construct the projection matrices for the two cameras
        Eigen::Matrix<double, 3 ,4> P1, P2;
        P1.block<3, 3>(0, 0) = rotation1;
        P1.block<3, 1>(0, 3) = translation1;
        //Eigen::Matrix<double, 3, 4> P1 = Eigen::Matrix<double, 3, 4>::Identity();  // The first camera is the reference camera
        //Eigen::Matrix<double, 3, 4> P2;
        P2.block<3, 3>(0, 0) = rotation2;
        P2.block<3, 1>(0, 3) = translation2;

        // Construct the coefficient matrix for the system of linear equations
        Eigen::Matrix4d A;
        A.row(0) = point1_h(0) * P1.row(2) - P1.row(0);
        A.row(1) = point1_h(1) * P1.row(2) - P1.row(1);
        A.row(2) = point2_h(0) * P2.row(2) - P2.row(0);
        A.row(3) = point2_h(1) * P2.row(2) - P2.row(1);

        // Solve the system of linear equations using SVD

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d X = svd.matrixV().col(3);

        /*
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(A.transpose() * A);
        Eigen::Vector4d X = eigensolver.eigenvectors().col(0);*/
        // Convert the result to inhomogeneous coordinates and return it
        return {X(0) / X(3), X(1) / X(3), X(2) / X(3)};
    }
};

