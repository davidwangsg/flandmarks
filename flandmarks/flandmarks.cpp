#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/generic_image.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        if (argc != 4) {
            cout << "Call this program like this:" << endl;
            cout << "./flandmarks model.dat inputImageList outputFile" << endl;
            return 0;
        }
        frontal_face_detector detector = get_frontal_face_detector();

        shape_predictor sp;
        deserialize(argv[1]) >> sp;

        std::string  inputImageList(argv[2]);
        std::string  outputFile(argv[3]);
        double image_size = 110;
        double image_padding = 0.3;

        std::string line, imagePath;
        std::ifstream infile(inputImageList.c_str());
        FILE* fp = fopen(outputFile.c_str(), "w");
        while (std::getline(infile, line))
        {
            imagePath = line;
            array2d<rgb_pixel> img;
            load_image(img, imagePath.c_str());

            // Make the image larger so we can detect small faces.
            long r1 = img.nr();
            pyramid_up(img);
            long r2 = img.nr();
            float scale = float(r2) / float(r1);

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.
            std::vector<rectangle> dets = detector(img);
            
            if(dets.size() == 0){
                continue;
            }else{
                cout<<line<<" #face="<<dets.size()<<endl;
               fprintf(fp, "%s;", line.c_str());
            }
            
            // find the largest face
            long max_area = -1;
            long max_area_idx = -1;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                long area = dets[j].area();
                if(area > max_area){
                    max_area = dets[j].area();
                    max_area_idx = j;
                }
            }
            std::vector<full_object_detection> shapes;
            full_object_detection shape = sp(img, dets[max_area_idx]);
            shapes.push_back(shape);

            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
            dlib::array<array2d<rgb_pixel> > face_chips;
            extract_image_chips(img, get_face_chip_details(shapes, image_size, image_padding), face_chips);

            for(int k = 0 ; k < shapes[0].num_parts()-1; k++){
                point& pt = shapes[0].part(k);
                fprintf(fp,"%.1f %.1f,", float(pt.x())/scale,float(pt.y())/scale);
            }
            int k = shapes[0].num_parts()-1;
            point& pt = shapes[0].part(k);
            fprintf(fp,"%.1f %.1f;0\n", float(pt.x())/scale,float(pt.y())/scale);
            
        }
        fclose(fp);
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}
