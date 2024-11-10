#include <ceres/ceres.h>
#include <Eigen/Core>
#include <vector>
#include <utility>
#include <ceres/rotation.h>

class BundleAdjustment {
public:
    explicit BundleAdjustment(std::vector<Eigen::Vector3d> points_3d,
                              const std::vector<Eigen::Vector2d>& points_2d,
                              Eigen::Matrix3d K,
                              Eigen::Vector3d& rotation_vector,
                              Eigen::Vector3d& translation_vector)
            : points_3d_(std::move(points_3d)),
              points_2d_(points_2d),
              K_(std::move(K)),
              rotation_vector_(rotation_vector),
              translation_vector_(translation_vector) {}

    std::pair<Eigen::Matrix3d, Eigen::Vector3d> GetCameraPose() {
        double rotation[3], translation[3];

        std::copy_n(rotation_vector_.data(), 3, rotation);
        std::copy_n(translation_vector_.data(), 3, translation);

        intrinsics_[0] = K_(0,0);
        intrinsics_[1] = K_(1,1);
        intrinsics_[2] = K_(0,2);
        intrinsics_[3] = K_(1,2);

        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.01); //modify the parameters here if not suitable

        for (size_t i = 0; i < points_2d_.size(); ++i) {
            ceres::CostFunction* cost_function = ReprojectionError::Create(points_2d_[i], points_3d_[i], intrinsics_);
            problem.AddResidualBlock(cost_function, loss_function, rotation, translation);
        }

        // Configure solver
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; //solver might be the problem. Also check the points correspondence
        // options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 300;
        options.function_tolerance = 1e-7;
        // options.min_line_search_step_size = 1e-10;
        options.parameter_tolerance = 1e-7;

        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        rotation_[0] = rotation[0];
        rotation_[1] = rotation[1];
        rotation_[2] = rotation[2];

        translation_[0] = translation[0];
        translation_[1] = translation[1];
        translation_[2] = translation[2];

        Eigen::Matrix3d rotation_matrix;
        ceres::AngleAxisToRotationMatrix(rotation, rotation_matrix.data());


        return { rotation_matrix, Eigen::Map<const Eigen::Vector3d>(translation)};
    }


    std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>> GetInliers(double max_error) const {
        std::vector<Eigen::Vector2d> points_2d_inliers;
        std::vector<Eigen::Vector3d> points_3d_inliers;

        for (size_t i = 0; i < points_2d_.size(); ++i) {
            Eigen::Vector2d projected = ProjectPoint(rotation_, translation_, points_3d_[i]);
            double error = (points_2d_[i] - projected).norm();
            // std::cout<<"error"<<error<<std::endl;
            if(error<=max_error){
                points_2d_inliers.emplace_back(points_2d_[i]);
                points_3d_inliers.emplace_back(points_3d_[i]);
            }
        }

        return {points_2d_inliers, points_3d_inliers};
    }

private:
    std::vector<Eigen::Vector3d> points_3d_;
    std::vector<Eigen::Vector2d> points_2d_;
    Eigen::Matrix3d K_;
    Eigen::Vector3d& rotation_vector_;
    Eigen::Vector3d& translation_vector_;
    double intrinsics_[4]; // [fx, fy, cx, cy]
    double rotation_[3]; // rotation vector (angle-axis representation)
    double translation_[3]; // translation vector

    struct ReprojectionError {
        ReprojectionError(const Eigen::Vector2d& observed, const Eigen::Vector3d& point, const double* intrinsics)
                : observed_(observed), point_(point), intrinsics_(intrinsics) {}

        template <typename T>
        bool operator()(const T* const camera_rotation, const T* const camera_translation, T* residuals) const {
            const Eigen::Matrix<T, 3, 1> point(static_cast<T>(point_(0)), static_cast<T>(point_(1)),
                                               static_cast<T>(point_(2)));
            Eigen::Matrix<T, 3, 1> rotated_point;

            // Use Ceres-provided angle-axis rotation function
            ceres::AngleAxisRotatePoint(camera_rotation, point.data(), rotated_point.data());
            // Translation
            rotated_point += Eigen::Matrix<T, 3, 1>(camera_translation[0], camera_translation[1], camera_translation[2]);

            // Project onto camera plane
            const T x_proj = rotated_point(0) / rotated_point(2);
            const T y_proj = rotated_point(1) / rotated_point(2);

            const T u = T(intrinsics_[0]) * x_proj + T(intrinsics_[2]);
            const T v = T(intrinsics_[1]) * y_proj + T(intrinsics_[3]);
            // Calculate residuals
            residuals[0] = T(observed_(0)) - u;
            residuals[1] = T(observed_(1)) - v;
            // std::cout<<residuals[0]<<std::endl;
            // std::cout<<residuals[1]<<std::endl;
            return true;
        }
        static ceres::CostFunction* Create(const Eigen::Vector2d& observed, const Eigen::Vector3d& point, const double* intrinsics) {
            return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                    new ReprojectionError(observed, point, intrinsics));
        }

        Eigen::Vector2d observed_;
        Eigen::Vector3d point_;
        const double* intrinsics_; // Added intrinsics to the cost function.
    };



        Eigen::Vector2d ProjectPoint(const double* rotation, const double* translation, const Eigen::Vector3d& point) const {
        // Assuming that the intrinsic parameters are represented as [f_x, f_y, c_x, c_y]
        const double& f_x = intrinsics_[0];
        const double& f_y = intrinsics_[1];
        const double& c_x = intrinsics_[2];
        const double& c_y = intrinsics_[3];

        double rotated_point[3];
        ceres::AngleAxisRotatePoint(rotation, point.data(), rotated_point);
        rotated_point[0] += translation[0];
        rotated_point[1] += translation[1];
        rotated_point[2] += translation[2];

        double x_proj = f_x * (rotated_point[0] / rotated_point[2]) + c_x;
        double y_proj = f_y * (rotated_point[1] / rotated_point[2]) + c_y;

        return {x_proj, y_proj};
    }




};
