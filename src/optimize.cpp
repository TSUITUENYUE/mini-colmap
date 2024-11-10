#include <utility>
#include <vector>
#include <fstream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

class Optimizer {
public:
    explicit Optimizer(
            const std::vector<Eigen::Vector2d>& points,
            Eigen::Matrix3d intrinsics,
            Eigen::Vector3d rotation,
            Eigen::Vector3d translation)
            : points_(points),
              intrinsics_(std::move(intrinsics)),
              rotation_(std::move(rotation)),
              translation_(std::move(translation)) {}

    std::pair<Eigen::Matrix3d, Eigen::Vector3d> optimize() {
        double a[3];
        a[0] = translation_[0];
        a[1] = translation_[1];
        a[2] = translation_[2];

        double b[3];
        b[0] = rotation_[0];
        b[1] = rotation_[1];
        b[2] = rotation_[2];

        double angle = rotation_.norm();
        Eigen::Vector3d axis = rotation_ / angle;
        Eigen::AngleAxisd angle_axis(angle, axis);
        Eigen::Matrix3d R = angle_axis.toRotationMatrix();
        Eigen::Vector3d T = translation_;

        std::vector<Eigen::Vector2d> points_2d = points_;
        std::vector<Eigen::Vector3d> points_3d;

        for (const auto& pts : points_2d) {
            Eigen::Matrix3d K_inv = intrinsics_.inverse();
            Eigen::Vector3d point_3d = unproject(pts, K_inv, R, T, 1.0);
            points_3d.emplace_back(point_3d);
        }

        ceres::Problem problem;

        for (size_t i = 0; i < points_2d.size(); ++i) {
            // Add cost function
            ceres::CostFunction* cost_function = ReprojectionError::Create(points_2d[i], points_3d[i]);
            problem.AddResidualBlock(cost_function, nullptr, b, a);
        }

        // Configure solver
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        // options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 200;

        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        Eigen::Vector3d new_translation(a[0], a[1], a[2]);
        double angle2 = Eigen::Vector3d(b[0], b[1], b[2]).norm();
        Eigen::Vector3d axis2 = Eigen::Vector3d(b[0], b[1], b[2]) / angle2;
        Eigen::AngleAxisd angle_axis2(angle2, axis2);
        Eigen::Matrix3d new_rotation = angle_axis2.toRotationMatrix();

        return std::make_pair(new_rotation, new_translation);
    }

private:
    struct ReprojectionError {
        ReprojectionError(const Eigen::Vector2d& observed, const Eigen::Vector3d& point)
                : observed_(observed), point_(point) {}

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

            // Calculate residuals
            residuals[0] = T(observed_(0)) - x_proj;
            residuals[1] = T(observed_(1)) - y_proj;

            return true;
        }

        static ceres::CostFunction* Create(const Eigen::Vector2d& observed, const Eigen::Vector3d& point) {
            return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                    new ReprojectionError(observed, point));
        }

        Eigen::Vector2d observed_;
        Eigen::Vector3d point_;
    };

    std::vector<Eigen::Vector2d> points_;
    Eigen::Matrix3d intrinsics_;
    Eigen::Vector3d rotation_;
    Eigen::Vector3d translation_;

    Eigen::Vector3d unproject(const Eigen::Vector2d& point, const Eigen::Matrix3d& K_inv,
                              const Eigen::Matrix3d& R, const Eigen::Vector3d& T, double depth) const {
        Eigen::Vector3d point_normalized = K_inv * Eigen::Vector3d(point(0), point(1), 1.0);
        Eigen::Vector3d point_3d = R.transpose() * (depth * point_normalized - T);
        return point_3d;
    }
};
