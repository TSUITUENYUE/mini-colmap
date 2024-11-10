#include <iostream>
#include <vector>
#include <colmap/base/database.h>
#include <Eigen/Core>
#include <fstream>
#include "read_data.cpp"
#include "estimate.cpp"
#include "optimize.cpp"
#include "triangulation.cpp"
#include "EPnP.cpp"
#include "bundle_adjust.cpp"

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

//this code can't work in specific environment (virtual machine eg).

void visualize(const std::vector<Eigen::Vector3d> &world_coord) {
    // Create vtkPoints object and add points into it
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    std::ofstream output("../output/points.csv");

    for (const auto& point : world_coord) {
        points->InsertNextPoint(point(0), point(1), point(2));
        output << point(0) << "," << point(1) << "," << point(2) << std::endl;
    }

    output.close();
    // Create a vtkPolyData object and add the points into it
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    // Create a mapper and actor
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polydata);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Add the actor to the scene
    renderer->AddActor(actor);
    renderer->SetBackground(0, 0, 0); // Background color white

    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();
    std::cout << "Press Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
}

/*
void visualize(const std::vector<Eigen::Vector3d> &world_coord) {
    // Create vtkPoints object and add points into it
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    std::ofstream output("../output/points.csv");

    for (const auto& point : world_coord) {
        points->InsertNextPoint(point(0), point(1), point(2));
        output << point(0) << "," << point(1) << "," << point(2) << std::endl;
    }

    output.close();

    // Create a vtkPolyData object and add the points into it
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    // Create a mapper and actor
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polydata);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Add the actor to the scene
    renderer->AddActor(actor);
    renderer->SetBackground(0, 0, 0); // Background color white

    // Render and interact
    renderWindow->Render();

    // Create a window to image filter and set the render window as its input
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->Update();

    // Create an image writer and set the filename and input image
    vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName("output.png");
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();

    std::cout << "Press Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
}
*/
bool find_element(std::vector<colmap::image_t>vec, colmap::image_t image_id){
    if (std::find(vec.begin(), vec.end(), image_id) != vec.end()) {
        return true;
    } else{
        return false;
    }
}

int locate(const std::vector<Eigen::Vector2d>& vec, const Eigen::Vector2d& item) {
    int index = -1;
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i].isApprox(item)) {
            index = i;
            //std::cout << index << std::endl;
            break;
        }
    }
    return index;
}



int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " /path/to/database.db "<< std::endl;
        return EXIT_FAILURE;
    }
    std::string database_path(argv[1]);
    DatabaseReader reader(database_path);
    std::vector<std::pair<colmap::image_pair_t, colmap::FeatureMatches>>all_matches = reader.ReadAllMatches();
    std::vector<colmap::Camera> all_cameras = reader.ReadAllCameras();
    std::vector<colmap::Image> all_images = reader.ReadAllImage();

    int iter_length = all_images.size();

    colmap::image_t image_id1, image_id2;
    colmap::image_pair_t image_pair_id;

    std::vector<colmap::image_t> image_pipeline;
    std::vector<std::pair<colmap::image_pair_t, colmap::FeatureMatches>> matches_pipeline;
    //decide first pipeline, find the most matched image_pair
    std::pair<colmap::image_pair_t, colmap::FeatureMatches> next_image_pair;
    int max =0;
    int maxi = 0;
    for(int i=0;i<all_matches.size();i++){
        if(all_matches[i].second.size() > max) {
            next_image_pair = all_matches[i];
            maxi = i;
            max = all_matches[i].second.size();
        }
    }
    all_matches.erase(all_matches.begin()+maxi);
    matches_pipeline.push_back(next_image_pair);

    image_pair_id = next_image_pair.first;
    colmap::Database::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);
    image_pipeline.push_back(image_id1);
    colmap::image_t last_image_id,new_image_id;
    last_image_id = image_id2;
    //decide the rest of pipeline
    //debug use
    // iter_length = 10;

    while(image_pipeline.size() != iter_length-1){
        int max =0;
        image_pipeline.push_back(last_image_id);
        for(int i=0;i<all_matches.size();i++){
            image_pair_id = all_matches[i].first;
            colmap::Database::PairIdToImagePair(image_pair_id, &image_id1, &image_id2);
            if(last_image_id == image_id1 and not find_element(image_pipeline,image_id2)){
                if(all_matches[i].second.size()>max){
                    next_image_pair = all_matches[i];
                    maxi = i;
                    max = all_matches[i].second.size();
                    new_image_id = image_id2;
                }
            }
            else if(last_image_id == image_id2 and not find_element(image_pipeline,image_id1)){
                if(all_matches[i].second.size()>max){
                    next_image_pair = all_matches[i];
                    maxi = i;
                    max = all_matches[i].second.size();
                    new_image_id = image_id1;
                }
            }
        }
        //std::cout<<last_image_id<<" "<<new_image_id<<endl;
        all_matches.erase(all_matches.begin()+maxi);
        matches_pipeline.push_back(next_image_pair);
        last_image_id = new_image_id;
        //image_pipeline.push_back(last_image_id);
    }
    image_pipeline.push_back(last_image_id);



    std::cout<<"start_reconstruction"<<endl;
    //start reconstruction
    std::vector<Eigen::Vector3d> world_coord;
    std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>> last_points_pair;
    std::vector<Eigen::Vector2d> last_points_2d;
    std::vector<Eigen::Vector3d> last_points_3d;
    std::unordered_map<std::string, Eigen::Vector3d> unique_points;
    std::vector<Eigen::Matrix4d>camera_poses;
    std::pair<Eigen::Matrix3d, Eigen::Vector3d> Last_RT;
    for(int i=0;i<iter_length-1;i++) {
        colmap::Database db(database_path);
        std::cout<<"iter_loop"<<i<<endl;
        Eigen::Matrix4d transformation_matrix;
        std::vector<Eigen::Vector3d> world_coord_temp;
        std::vector<Eigen::Vector3d> world_coord_valid;
        std::vector<Eigen::Vector2d> points2d_valid;
        //load pipeline
        next_image_pair = matches_pipeline[i];
        /*
        image_id1 = image_pipeline[i];
        image_id2 = image_pipeline[i+1]; */
        colmap::Database::PairIdToImagePair(next_image_pair.first, &image_id1, &image_id2);
        std::cout<<image_id1<<" "<<image_id2<<endl;
        colmap::Image image1 = reader.ReadImage(image_id1);
        colmap::Image image2 = reader.ReadImage(image_id2);
        colmap::Camera camera1 = reader.ReadCamera(image1.CameraId());
        colmap::Camera camera2 = reader.ReadCamera(image2.CameraId());
        Estimator estimator(db, image_id1, image_id2, next_image_pair.second);

        if(i == 0){
            //initialization with the first loop, use two_view_geometry
            estimator.run();
            //get the base rotation vector and translation vector using two view geometry
            Eigen::Vector3d rotation_vector = estimator.getRotationVector();
            Eigen::Vector3d translation_vector = estimator.getTranslationVector();
            //get keypoint from two images
            std::vector<Eigen::Vector2d> points1 = estimator.getKeyPoints().first;
            std::vector<Eigen::Vector2d> points2 = estimator.getKeyPoints().second;
            //points1.assign(points1.begin(), points1.begin() + 500);
            //points2.assign(points2.begin(), points2.begin() + 500);
            //get intrinsics
            Eigen::Matrix3d K = reader.GetCalibrationMatrix(camera2);

            // std::cout<<K(0,0)<<" "<<K(1,1)<<" "<<K(0,2)<<" "<<K(1,2)<<endl;

            Eigen::Matrix3d R = Eigen::AngleAxisd(rotation_vector.norm(), rotation_vector.normalized()).toRotationMatrix();
            Eigen::Vector3d T = translation_vector;
            Eigen::Matrix3d Last_R = Eigen::Matrix<double, 3, 3>::Identity();
            Eigen::Vector3d Last_T(0,0,0);
            //triangulation for the first set of points
            Triangulation triangulation(points1, points2, K, Last_R, Last_T, R, T);
            //initialize world coord
            triangulation.run();
            last_points_pair = triangulation.getPointsPair();
            world_coord_temp = last_points_pair.second;
            /*
            for(int j =0;j<world_coord_temp.size();j++){
                std::cout<<world_coord_temp[j]<<endl;
            }
            */
            for(int j = 0;j<world_coord_temp.size();j++){
                if(world_coord_temp[j].norm()<1e1 and world_coord_temp[j].norm()>1e-1){
                    world_coord_valid.emplace_back(world_coord_temp[j]);
                    points2d_valid.emplace_back(points2[j]);
                }
            }
            // std::cout<<points2d_valid.size()<<endl;
            // std::cout<<world_coord_valid.size()<<endl;

            last_points_pair = std::make_pair(points2d_valid, world_coord_valid);

            BundleAdjustment lba(world_coord_valid,points2d_valid,K,rotation_vector,T);
            std::pair<Eigen::Matrix3d, Eigen::Vector3d> RT = lba.GetCameraPose();
            R = RT.first;
            T = RT.second;
            last_points_pair = lba.GetInliers(1.0);
            world_coord_temp=last_points_pair.second;
            transformation_matrix = Eigen::Matrix4d::Identity();
            transformation_matrix.block<3,3>(0,0) = R;
            transformation_matrix.block<3,1>(0,3) = T;
            Last_RT = RT;

            /*
            for(int k=0;k<3;k++){
                std::cout<<R(k,0)<<R(k,1)<<R(k,2)<<endl;
            }
            std::cout<<T(0)<<" "<<T(1)<<" "<<T(2)<<endl;*/
        }
        else{
            estimator.run_next();
            //get keypoint from two images
            std::vector<Eigen::Vector2d> points1 = estimator.getKeyPoints().first;
            std::vector<Eigen::Vector2d> points2 = estimator.getKeyPoints().second;
            //points1.assign(points1.begin(), points1.begin() + 500);
            //points2.assign(points2.begin(), points2.begin() + 500);
            last_points_2d = last_points_pair.first;
            last_points_3d = last_points_pair.second;
            // std::cout<<points1.size()<<endl;
            int count1 = 0;
            int count2 = 0;
            for (int j = 0; j < points1.size(); j++) {
                int index1 = locate(last_points_2d, points1[j]);
                int index2 = locate(last_points_2d, points2[j]);
                if (index1 != -1) {
                    count1++;
                }
                if (index2 != -1) {
                    count2++;
                }
            }
            // std::cout<<count1<<count2<<endl;
            bool flag = count1>count2;
            //more count means old points
            std::vector<Eigen::Vector2d> old_points_2d, this_round_points;

            if(flag){
                old_points_2d = points1;
                this_round_points = points2;
            }
            else{
                old_points_2d = points2;
                this_round_points = points1;
            }

            Eigen::Matrix3d K = reader.GetCalibrationMatrix(camera2);

            std::vector<Eigen::Vector2d> new_points_2d, first8_2d;
            std::vector<Eigen::Vector3d> new_points_3d, first8_3d;
            //std::cout<<old_points_2d.size()<<endl;
            int index = -1;
            for (int j = 0;j<old_points_2d.size();j++){
                index = locate(last_points_2d, old_points_2d[j]);
                // std::cout<<index<<endl;
                if(index >= 0){
                    new_points_2d.push_back(this_round_points[j]);
                    // locate the correspondent 3d points
                    new_points_3d.push_back(last_points_3d[index]);
                }
            }

            /*
            for (int j = 0;j<new_points_2d.size();j++){
                std::cout<<new_points_2d[j]<<endl;
                std::cout<<new_points_3d[j]<<endl;
            }*/

            // std::cout<<new_points_2d.size()<<endl;
            // std::cout<<new_points_3d.size()<<endl;
            //use PnP to estimate the new camera pose
            first8_2d.assign(new_points_2d.begin(), new_points_2d.begin() + 6);
            first8_3d.assign(new_points_3d.begin(), new_points_3d.begin() + 6);


            Eigen::Matrix3d K_inv = K.inverse();
            for (int i = 0; i < first8_2d.size(); ++i) {
                first8_2d[i] = (K_inv * first8_2d[i].homogeneous()).hnormalized();
            }

            EPnP epnp(first8_3d, first8_2d, K);
            // std::cout<<"epnp finished"<<endl;
            std::pair<Eigen::Matrix3d, Eigen::Vector3d>RT =epnp.getCameraPose();
            // std::cout<<"pnp"<<endl;
            Eigen::Matrix3d R = RT.first;
            Eigen::Vector3d T = RT.second;


            /*std::cout<<"Before"<<endl;
            for(int k=0;k<3;k++){
                std::cout<<R(k,0)<<R(k,1)<<R(k,2)<<endl;
            }
            std::cout<<T(0)<<" "<<T(1)<<" "<<T(2)<<endl;*/
            Eigen::Vector3d rotation_vector;
            ceres::RotationMatrixToAngleAxis(R.data(), rotation_vector.data());
            Eigen::Matrix3d Last_R = Last_RT.first;
            Eigen::Vector3d Last_T = Last_RT.second;

            Triangulation triangulation(old_points_2d, this_round_points, K, Last_R, Last_T, R, T);
            triangulation.run();
            // last_points_pair = triangulation.getPointsPair();
            world_coord_temp = triangulation.getWorldCoord();
            for(int j = 0;j<world_coord_temp.size();j++){
                if(world_coord_temp[j].norm()<1e1 and world_coord_temp[j].norm()>1e-1){
                    world_coord_valid.emplace_back(world_coord_temp[j]);
                    points2d_valid.emplace_back(this_round_points[j]);
                }
            }
            // std::cout<<points2d_valid.size()<<endl;
            // std::cout<<world_coord_valid.size()<<endl;

            last_points_pair = std::make_pair(points2d_valid, world_coord_valid);

            this_round_points = last_points_pair.first;
            world_coord_temp = last_points_pair.second;

            BundleAdjustment lba(world_coord_temp,this_round_points,K,rotation_vector,T);
            //lba.run();
            RT=lba.GetCameraPose();
            R=RT.first;
            T=RT.second;
            // last_points_pair = lba.GetInliers(2.0);
            Last_RT = RT;

            world_coord_temp = last_points_pair.second;
            // std::cout<<world_coord_temp.size()<<endl;
            //world_coord_temp=lba.GetPoints3D();
            /*
            std::cout<<"After"<<endl;
            for(int k=0;k<3;k++){
                std::cout<<R(k,0)<<R(k,1)<<R(k,2)<<endl;
            }
            std::cout<<T(0)<<" "<<T(1)<<" "<<T(2)<<endl;*/

            transformation_matrix = Eigen::Matrix4d::Identity();
            transformation_matrix.block<3,3>(0,0) = R;
            transformation_matrix.block<3,1>(0,3) = T;
        }
        camera_poses.push_back(transformation_matrix);
        //check unique points
        for (const auto& p : world_coord_temp) {
            std::ostringstream oss;
            oss << p.x() << " " << p.y() << " " << p.z();
            std::string point_string = oss.str();

            // if the point is not in the map yet, add it to both the map and the world_coord vector
            if (unique_points.find(point_string) == unique_points.end()) {
                unique_points[point_string] = p;
                world_coord.push_back(p);
            }
        }
    }
    visualize(world_coord);
}

