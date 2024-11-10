#include <utility>
#include <vector>
#include <fstream>
#include <colmap/base/database.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
class EPnP {
public:
    explicit EPnP(
            const std::vector<Eigen::Vector3d>& points3d,
            const std::vector<Eigen::Vector2d>& points2d,
            Eigen::Matrix3d intrinsics)
            : points3d_(&points3d), points2d_(&points2d), K_(std::move(intrinsics)) {}

    std::pair<Eigen::Matrix3d, Eigen::Vector3d> getCameraPose() {
        ChooseControlPoints();
        // std::cout<<cws_[0]<<" "<<cws_[1]<<" "<<cws_[2]<<" "<<cws_[3]<<std::endl;
        ComputeBarycentricCoordinates();
        // std::cout<<"here"<<std::endl;
        // std::cout<<alphas_[0]<<alphas_[1]<<alphas_[2]<<alphas_[3]<<std::endl;
        const Eigen::Matrix<double, Eigen::Dynamic, 12> M = ComputeM();
        const Eigen::Matrix<double, 12, 12> MtM = M.transpose() * M;

        Eigen::JacobiSVD<Eigen::Matrix<double, 12, 12>> svd(
                MtM, Eigen::ComputeFullV | Eigen::ComputeFullU);
        const Eigen::Matrix<double, 12, 12> Ut = svd.matrixU().transpose();

        const Eigen::Matrix<double, 6, 10> L6x10 = ComputeL6x10(Ut);
        const Eigen::Matrix<double, 6, 1> rho = ComputeRho();

        Eigen::Vector4d betas[4];
        std::array<double, 4> reproj_errors;
        std::array<Eigen::Matrix3d, 4> Rs;
        std::array<Eigen::Vector3d, 4> ts;
        // std::cout<<reproj_errors[0]<<" "<<reproj_errors[1]<<" "<<reproj_errors[2]<<" "<<reproj_errors[3]<<std::endl;
        FindBetasApprox1(L6x10, rho, &betas[1]);
        RunGaussNewton(L6x10, rho, &betas[1]);
        reproj_errors[1] = ComputeRT(Ut, betas[1], &Rs[1], &ts[1]);
        FindBetasApprox2(L6x10, rho, &betas[2]);
        RunGaussNewton(L6x10, rho, &betas[2]);
        reproj_errors[2] = ComputeRT(Ut, betas[2], &Rs[2], &ts[2]);

        FindBetasApprox3(L6x10, rho, &betas[3]);
        RunGaussNewton(L6x10, rho, &betas[3]);
        reproj_errors[3] = ComputeRT(Ut, betas[3], &Rs[3], &ts[3]);


        int best_idx = 1;
        if (reproj_errors[2] < reproj_errors[1]) {
            best_idx = 2;
        }
        if (reproj_errors[3] < reproj_errors[best_idx]) {
            best_idx = 3;
        }

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        R = Rs[best_idx];
        T = ts[best_idx];
        return {R, T};
    }

private:

    const std::vector<Eigen::Vector2d>* points2d_ = nullptr;

    const std::vector<Eigen::Vector3d>* points3d_ = nullptr;
    std::vector<Eigen::Vector3d> pcs_;
    std::vector<Eigen::Vector4d> alphas_;
    std::array<Eigen::Vector3d, 4> cws_;
    std::array<Eigen::Vector3d, 4> ccs_;
    Eigen::Matrix3d K_;

    void ChooseControlPoints() {
        // Take C0 as the reference points centroid:
        Eigen::Vector3d Cw(0, 0, 0);
        cws_[0] = Cw;
        for (size_t i = 0; i < points3d_->size(); ++i) {
            cws_[0] += (*points3d_)[i];
        }
        cws_[0] = cws_[0] / points3d_->size();
        // std::cout<<cws_[0]<<std::endl;
        Eigen::Matrix<double, Eigen::Dynamic, 3> PW0(points3d_->size(), 3);
        for (size_t i = 0; i < points3d_->size(); ++i) {
            PW0.row(i) = (*points3d_)[i] - cws_[0];
        }

        const Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                PW0tPW0, Eigen::ComputeFullV | Eigen::ComputeFullU);
        const Eigen::Vector3d D = svd.singularValues();
        const Eigen::Matrix3d Ut = svd.matrixU().transpose();

        for (int i = 1; i < 4; ++i) {
            const double k = std::sqrt(D(i - 1) / points3d_->size());
            cws_[i] = cws_[0] + k * Ut.row(i - 1).transpose();
        }
    }

    bool ComputeBarycentricCoordinates() {
        Eigen::Matrix3d CC;
        for (int i = 0; i < 3; ++i) {
            for (int j = 1; j < 4; ++j) {
                CC(i, j - 1) = cws_[j][i] - cws_[0][i];
            }
        }

        if (CC.colPivHouseholderQr().rank() < 3) {
            return false;}

        const Eigen::Matrix3d CC_inv = CC.inverse();

        alphas_.resize(points2d_->size());
        for (size_t i = 0; i < points3d_->size(); ++i) {
            for (int j = 0; j < 3; ++j) {
                alphas_[i][1 + j] = CC_inv(j, 0) * ((*points3d_)[i][0] - cws_[0][0]) +
                                    CC_inv(j, 1) * ((*points3d_)[i][1] - cws_[0][1]) +
                                    CC_inv(j, 2) * ((*points3d_)[i][2] - cws_[0][2]);
            }
            alphas_[i][0] = 1.0 - alphas_[i][1] - alphas_[i][2] - alphas_[i][3];
        }

        return true;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 12> ComputeM() {
        Eigen::Matrix<double, Eigen::Dynamic, 12> M(2 * points2d_->size(), 12);
        // Eigen::Matrix3d K_inv = K_.inverse();
        for (size_t i = 0; i < points3d_->size(); ++i) {

            for (size_t j = 0; j < 4; ++j) {
                M(2 * i, 3 * j) = alphas_[i][j];
                M(2 * i, 3 * j + 1) = 0.0;
                M(2 * i, 3 * j + 2) = -alphas_[i][j] * (*points2d_)[i].x();

                M(2 * i + 1, 3 * j) = 0.0;
                M(2 * i + 1, 3 * j + 1) = alphas_[i][j];
                M(2 * i + 1, 3 * j + 2) = -alphas_[i][j] * (*points2d_)[i].y();
            }
        }
        return M;
    }

    Eigen::Matrix<double , 6, 10> ComputeL6x10(const Eigen::Matrix<double , 12, 12>& Ut) {
        Eigen::Matrix<double , 6, 10> L6x10;
        std::array<std::array<Eigen::Vector3d , 6>, 4> dv;
        for (int i = 0; i < 4; ++i) {
            int a = 0, b = 1;
            for (int j = 0; j < 6; ++j) {
                dv[i][j][0] = Ut(11 - i, 3 * a) - Ut(11 - i, 3 * b);
                dv[i][j][1] = Ut(11 - i, 3 * a + 1) - Ut(11 - i, 3 * b + 1);
                dv[i][j][2] = Ut(11 - i, 3 * a + 2) - Ut(11 - i, 3 * b + 2);
                b += 1;
                if (b > 3) {
                    a += 1;
                    b = a + 1;
                     }
                }
            }

        for (int i = 0; i < 6; ++i) {
            L6x10(i, 0) = dv[0][i].transpose() * dv[0][i];
            L6x10(i, 1) = 2.0 * dv[0][i].transpose() * dv[1][i];
            L6x10(i, 2) = dv[1][i].transpose() * dv[1][i];
            L6x10(i, 3) = 2.0 * dv[0][i].transpose() * dv[2][i];
            L6x10(i, 4) = 2.0 * dv[1][i].transpose() * dv[2][i];
            L6x10(i, 5) = dv[2][i].transpose() * dv[2][i];
            L6x10(i, 6) = 2.0 * dv[0][i].transpose() * dv[3][i];
            L6x10(i, 7) = 2.0 * dv[1][i].transpose() * dv[3][i];
            L6x10(i, 8) = 2.0 * dv[2][i].transpose() * dv[3][i];
            L6x10(i, 9) = dv[3][i].transpose() * dv[3][i];
            }

        return L6x10;
        }

    Eigen::Matrix<double , 6, 1> ComputeRho() {
        Eigen::Matrix<double , 6, 1> rho;
        rho[0] = (cws_[0] - cws_[1]).squaredNorm();
        rho[1] = (cws_[0] - cws_[2]).squaredNorm();
        rho[2] = (cws_[0] - cws_[3]).squaredNorm();
        rho[3] = (cws_[1] - cws_[2]).squaredNorm();
        rho[4] = (cws_[1] - cws_[3]).squaredNorm();
        rho[5] = (cws_[2] - cws_[3]).squaredNorm();
        return rho;
    }

    void FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L6x10,
                                         const Eigen::Matrix<double, 6, 1>& rho,
                                         Eigen::Vector4d* betas) {
        Eigen::Matrix<double, 6, 4> L_6x4;
        for (int i = 0; i < 6; ++i) {
            L_6x4(i, 0) = L6x10(i, 0);
            L_6x4(i, 1) = L6x10(i, 1);
            L_6x4(i, 2) = L6x10(i, 3);
            L_6x4(i, 3) = L6x10(i, 6);
        }

        Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(
                L_6x4, Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::Matrix<double, 6, 1> Rho_temp = rho;
        const Eigen::Matrix<double, 4, 1> b4 = svd.solve(Rho_temp);

        if (b4[0] < 0) {
            (*betas)[0] = std::sqrt(-b4[0]);
            (*betas)[1] = -b4[1] / (*betas)[0];
            (*betas)[2] = -b4[2] / (*betas)[0];
            (*betas)[3] = -b4[3] / (*betas)[0];
        } else {
            (*betas)[0] = std::sqrt(b4[0]);
            (*betas)[1] = b4[1] / (*betas)[0];
            (*betas)[2] = b4[2] / (*betas)[0];
            (*betas)[3] = b4[3] / (*betas)[0];
        }
    }

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

    void FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L6x10,
                                         const Eigen::Matrix<double, 6, 1>& rho,
                                         Eigen::Vector4d* betas) {
        Eigen::Matrix<double, 6, 3> L_6x3(6, 3);

        for (int i = 0; i < 6; ++i) {
            L_6x3(i, 0) = L6x10(i, 0);
            L_6x3(i, 1) = L6x10(i, 1);
            L_6x3(i, 2) = L6x10(i, 2);
        }

        Eigen::JacobiSVD<Eigen::Matrix<double, 6, 3>> svd(
                L_6x3, Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::Matrix<double, 6, 1> Rho_temp = rho;
        const Eigen::Matrix<double, 3, 1> b3 = svd.solve(Rho_temp);

        if (b3[0] < 0) {
            (*betas)[0] = std::sqrt(-b3[0]);
            (*betas)[1] = (b3[2] < 0) ? std::sqrt(-b3[2]) : 0.0;
        } else {
            (*betas)[0] = std::sqrt(b3[0]);
            (*betas)[1] = (b3[2] > 0) ? std::sqrt(b3[2]) : 0.0;
        }

        if (b3[1] < 0) {
            (*betas)[0] = -(*betas)[0];
        }

        (*betas)[2] = 0.0;
        (*betas)[3] = 0.0;
    }

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

    void FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L6x10,
                                         const Eigen::Matrix<double, 6, 1>& rho,
                                         Eigen::Vector4d* betas) {
        Eigen::JacobiSVD<Eigen::Matrix<double, 6, 5>> svd(
                L6x10.leftCols<5>(), Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::Matrix<double, 6, 1> Rho_temp = rho;
        const Eigen::Matrix<double, 5, 1> b5 = svd.solve(Rho_temp);

        if (b5[0] < 0) {
            (*betas)[0] = std::sqrt(-b5[0]);
            (*betas)[1] = (b5[2] < 0) ? std::sqrt(-b5[2]) : 0.0;
        } else {
            (*betas)[0] = std::sqrt(b5[0]);
            (*betas)[1] = (b5[2] > 0) ? std::sqrt(b5[2]) : 0.0;
        }
        if (b5[1] < 0) {
            (*betas)[0] = -(*betas)[0];
        }
        (*betas)[2] = b5[3] / (*betas)[0];
        (*betas)[3] = 0.0;
    }

    void RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L6x10,
                                       const Eigen::Matrix<double, 6, 1>& rho,
                                       Eigen::Vector4d* betas) {
        Eigen::Matrix<double, 6, 4> A;
        Eigen::Matrix<double, 6, 1> b;

        const int kNumIterations = 5;
        for (int k = 0; k < kNumIterations; ++k) {
            for (int i = 0; i < 6; ++i) {
                A(i, 0) = 2 * L6x10(i, 0) * (*betas)[0] + L6x10(i, 1) * (*betas)[1] +
                          L6x10(i, 3) * (*betas)[2] + L6x10(i, 6) * (*betas)[3];
                A(i, 1) = L6x10(i, 1) * (*betas)[0] + 2 * L6x10(i, 2) * (*betas)[1] +
                          L6x10(i, 4) * (*betas)[2] + L6x10(i, 7) * (*betas)[3];
                A(i, 2) = L6x10(i, 3) * (*betas)[0] + L6x10(i, 4) * (*betas)[1] +
                          2 * L6x10(i, 5) * (*betas)[2] + L6x10(i, 8) * (*betas)[3];
                A(i, 3) = L6x10(i, 6) * (*betas)[0] + L6x10(i, 7) * (*betas)[1] +
                          L6x10(i, 8) * (*betas)[2] + 2 * L6x10(i, 9) * (*betas)[3];

                b(i) = rho[i] - (L6x10(i, 0) * (*betas)[0] * (*betas)[0] +
                                 L6x10(i, 1) * (*betas)[0] * (*betas)[1] +
                                 L6x10(i, 2) * (*betas)[1] * (*betas)[1] +
                                 L6x10(i, 3) * (*betas)[0] * (*betas)[2] +
                                 L6x10(i, 4) * (*betas)[1] * (*betas)[2] +
                                 L6x10(i, 5) * (*betas)[2] * (*betas)[2] +
                                 L6x10(i, 6) * (*betas)[0] * (*betas)[3] +
                                 L6x10(i, 7) * (*betas)[1] * (*betas)[3] +
                                 L6x10(i, 8) * (*betas)[2] * (*betas)[3] +
                                 L6x10(i, 9) * (*betas)[3] * (*betas)[3]);
            }

            const Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

            (*betas) += x;
        }
    }

    double ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut,
                                    const Eigen::Vector4d& betas,
                                    Eigen::Matrix3d* R, Eigen::Vector3d* t) {
        ComputeCcs(betas, Ut);
        ComputePcs();
        SolveForSign();
        EstimateRT(R, t);
        return ComputeTotalReprojectionError(*R, *t);
    }

    void ComputeCcs(const Eigen::Vector4d& betas,
                                   const Eigen::Matrix<double, 12, 12>& Ut) {
        for (int i = 0; i < 4; ++i) {
            ccs_[i][0] = ccs_[i][1] = ccs_[i][2] = 0.0;
        }

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 3; ++k) {
                    ccs_[j][k] += betas[i] * Ut(11 - i, 3 * j + k);
                }
            }
        }
    }


    void ComputePcs() {
        pcs_.resize(points2d_->size());
        for (size_t i = 0; i < points3d_->size(); ++i) {
            for (int j = 0; j < 3; ++j) {
                pcs_[i][j] = alphas_[i][0] * ccs_[0][j] + alphas_[i][1] * ccs_[1][j] +
                             alphas_[i][2] * ccs_[2][j] + alphas_[i][3] * ccs_[3][j];
            }
        }
    }

    void SolveForSign() {
        if (pcs_[0][2] < 0.0) {
            for (int i = 0; i < 4; ++i) {
                ccs_[i] = -ccs_[i];
            }
            for (size_t i = 0; i < points3d_->size(); ++i) {
                pcs_[i] = -pcs_[i];
            }
        }
    }

    void EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t) {
        Eigen::Vector3d pc0 = Eigen::Vector3d::Zero();
        Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();

        for (size_t i = 0; i < points3d_->size(); ++i) {
            pc0 += pcs_[i];
            pw0 += (*points3d_)[i];
        }
        pc0 /= points3d_->size();
        pw0 /= points3d_->size();

        Eigen::Matrix3d abt = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < points3d_->size(); ++i) {
            for (int j = 0; j < 3; ++j) {
                abt(j, 0) += (pcs_[i][j] - pc0[j]) * ((*points3d_)[i][0] - pw0[0]);
                abt(j, 1) += (pcs_[i][j] - pc0[j]) * ((*points3d_)[i][1] - pw0[1]);
                abt(j, 2) += (pcs_[i][j] - pc0[j]) * ((*points3d_)[i][2] - pw0[2]);
            }
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                abt, Eigen::ComputeFullV | Eigen::ComputeFullU);
        const Eigen::Matrix3d abt_U = svd.matrixU();
        const Eigen::Matrix3d abt_V = svd.matrixV();

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                (*R)(i, j) = abt_U.row(i) * abt_V.row(j).transpose();
            }
        }

        if (R->determinant() < 0) {
            Eigen::Matrix3d Abt_v_prime = abt_V;
            Abt_v_prime.col(2) = -abt_V.col(2);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    (*R)(i, j) = abt_U.row(i) * Abt_v_prime.row(j).transpose();
                }
            }
        }

        *t = pc0 - *R * pw0;
    }
    void ComputeSquaredReprojectionError(
            const std::vector<Eigen::Vector2d>& points2D,
            const std::vector<Eigen::Vector3d>& points3D,
            const Eigen::Matrix3x4d& proj_matrix, std::vector<double>* residuals) {
        CHECK_EQ(points2D.size(), points3D.size());

        residuals->resize(points2D.size());

        // Note that this code might not be as nice as Eigen expressions,
        // but it is significantly faster in various tests.

        const double P_00 = proj_matrix(0, 0);
        const double P_01 = proj_matrix(0, 1);
        const double P_02 = proj_matrix(0, 2);
        const double P_03 = proj_matrix(0, 3);
        const double P_10 = proj_matrix(1, 0);
        const double P_11 = proj_matrix(1, 1);
        const double P_12 = proj_matrix(1, 2);
        const double P_13 = proj_matrix(1, 3);
        const double P_20 = proj_matrix(2, 0);
        const double P_21 = proj_matrix(2, 1);
        const double P_22 = proj_matrix(2, 2);
        const double P_23 = proj_matrix(2, 3);

        for (size_t i = 0; i < points2D.size(); ++i) {
            const double X_0 = points3D[i](0);
            const double X_1 = points3D[i](1);
            const double X_2 = points3D[i](2);

            // Project 3D point from world to camera.
            const double px_2 = P_20 * X_0 + P_21 * X_1 + P_22 * X_2 + P_23;

            // Check if 3D point is in front of camera.
            if (px_2 > std::numeric_limits<double>::epsilon()) {
                const double px_0 = P_00 * X_0 + P_01 * X_1 + P_02 * X_2 + P_03;
                const double px_1 = P_10 * X_0 + P_11 * X_1 + P_12 * X_2 + P_13;

                const double x_0 = points2D[i](0);
                const double x_1 = points2D[i](1);

                const double inv_px_2 = 1.0 / px_2;
                const double dx_0 = x_0 - px_0 * inv_px_2;
                const double dx_1 = x_1 - px_1 * inv_px_2;

                (*residuals)[i] = dx_0 * dx_0 + dx_1 * dx_1;
            } else {
                (*residuals)[i] = std::numeric_limits<double>::max();
            }
        }
    }


    double ComputeTotalReprojectionError(const Eigen::Matrix3d& R,
                                                        const Eigen::Vector3d& t) {
        Eigen::Matrix3x4d proj_matrix;
        proj_matrix.leftCols<3>() = R;
        proj_matrix.rightCols<1>() = t;

        std::vector<double> residuals;
        ComputeSquaredReprojectionError(*points2d_, *points3d_, proj_matrix,
                                        &residuals);

        double reproj_error = 0.0;
        for (const double residual : residuals) {
            reproj_error += std::sqrt(residual);
        }

        return reproj_error;


    }
};