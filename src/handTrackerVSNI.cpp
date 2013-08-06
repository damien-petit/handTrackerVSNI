#include "handTrackerVSNI.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>
#include <fstream>
#include <GL/gl.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <ctime>

#include <visionsystem/vs_plugins/xmlrpc/xmlrpc-server.h>

#include <time.h>

#include <visionsystem/vs_controllers/openni/cameraopenni.h>

inline void time_diff(const timeval & tv_in, const timeval & tv_out, timeval & tv_diff)
{
    if( tv_out.tv_sec < tv_in.tv_sec || (tv_out.tv_sec == tv_in.tv_sec && tv_out.tv_usec < tv_in.tv_usec) )
    {
        time_diff(tv_out, tv_in, tv_diff);
    }
    else
    {
        if(tv_out.tv_usec < tv_in.tv_usec)
        {
            tv_diff.tv_sec = tv_out.tv_sec - tv_in.tv_sec - 1;
            tv_diff.tv_usec = 1000000 - tv_in.tv_usec + tv_out.tv_usec;
        }
        else
        {
            tv_diff.tv_sec = tv_out.tv_sec - tv_in.tv_sec;
            tv_diff.tv_usec = tv_out.tv_usec - tv_in.tv_usec;
        }
    }
}

#define TIME_CALL(x) \
{ \
timeval tv_in;\
timeval tv_out;\
timeval tv_diff;\
gettimeofday(&tv_in, 0);\
x;\
gettimeofday(&tv_out, 0);\
time_diff(tv_in, tv_out, tv_diff);\
std::cout << "Call " << #x << " took: " << (tv_diff.tv_sec + ((float)tv_diff.tv_usec/1000000)) << "s" << std::endl;\
}

namespace handTrackerVSNI
{
//used to access the member of the class handTrackerVSNI inside the static callback function of OpenNI
    HandTrackerVSNI* me_ = 0;

    HandTrackerVSNI::HandTrackerVSNI( visionsystem::VisionSystem *vs, string sandbox )
    : Plugin( vs, "handTrackerVSNI", sandbox ), WithViewer(vs),
    XmlRpcServerMethod("GetObjectPosition", 0),
    imgXi_(0), imgXd_(0),imgDispXi_(0), imgDispXd_(0),
    sandbox_(sandbox)
    {
        me_ = this;

        cv::FileStorage dataToLoadOpenCV;

        if(!dataToLoadOpenCV.open(sandbox +"dataToLoadOpenCVFile.yml", cv::FileStorage::READ))
        {
            std::cout<<"ERROR during the opening of the file dataCalibCameraLeftOpenCV.yml"<<std::endl;
        }

        //initialisation of the different camera matrix calibration        

        dataToLoadOpenCV["cameraMatrixXtionRGB"] >> cameraMatrixXtionRGB_;
        dataToLoadOpenCV["distorsionMatrixXtionRGB"] >> distorsionMatrixXtionRGB_;
        dataToLoadOpenCV["headToCamMatrixXtionRGB"] >> headToCamMatrixXtionRGB_;

//        std::cout<<cameraMatrixXtionRGB_<<std::endl;
//        std::cout<<headToCamMatrixXtionRGB_<<std::endl;

        dataToLoadOpenCV["coshellName_"] >> coshellName_;

        dataToLoadOpenCV.release();

// used in the resul;t transmission function by xmlrpc


        std::cout<<"coshellName_ "<<coshellName_<<std::endl;

        coshellVision_ = new coshell::CoshellClient(coshellName_, 2809, false);
        //We have to  initialize the coshell since we dont use coshell interpreter
        coshellVision_->Initialize();

        NIHandMatMono_.create(480,640, CV_8UC1);
//        nUsers_ = 15;
    }

    HandTrackerVSNI::~HandTrackerVSNI()
    {
        delete imgXi_;
        delete imgDispXi_;
        delete imgXd_;
        delete imgDispXd_;
        delete coshellVision_;
    }

    inline void rgb_to_nb(const vision::Image<uint32_t, vision::RGB> & image, vision::Image<unsigned char, vision::MONO> & imageNB)
    {
        uint32_t rgb = 0;
        for(unsigned int i=0; i< image.pixels; ++i)
        {
            rgb = image.raw_data[i];
            imageNB.raw_data[i] =  (unsigned char)( ( ((rgb & 0xFF0000) >> 16)+ ((rgb & 0x00FF00) >> 8) + (rgb & 0x0000FF) )/3 ) ;
        }
    }

    void HandTrackerVSNI::parse_config_line( vector<string> &line )
    {
        if( fill_member(line, "xtion-image", camNameXi_) )
            return;
        if( fill_member(line, "xtion-depth", camNameXd_) )
            return;
    }

    bool HandTrackerVSNI::pre_fct()
    {
        std::cout << "[handTrackerVSNI] pre_fct()" << endl ;
        string filename = get_sandbox() + string("/handTrackerVSNI.conf") ;

        try 
        {
            read_config_file ( filename.c_str() ) ;
        } 
        catch ( string msg ) 
        {
            throw(std::string("handTrackerVSNI will not work without a correct handTrackerVSNI.conf config file"));
        }

        std::cout<<" camNameXi_ "<<camNameXi_<<std::endl;
        std::cout<<" camNameXd_ "<<camNameXd_<<std::endl;

        camXi_ = get_camera(camNameXi_);
        camXd_ = get_camera(camNameXd_);

        if(camXi_ == 0 || camXd_ == 0)
        {
            throw(std::string("handTrackerVSNI expect one image camera and one depth camera"));
        }

        register_to_cam< vision::Image<uint32_t, vision::RGB> >(camXi_, 100);
        imgXi_ = new vision::Image<uint32_t, vision::RGB>(camXi_->get_size());
        imgDispXi_ = new vision::Image<uint32_t, vision::RGB>(camXi_->get_size());

        imgXiMONO_ = new vision::Image<uint8_t, vision::MONO>(camXi_->get_size());
//        NiHandMatImgMono_ = cvCreateImage(cvSize(imgXi_->width,imgXi_->height), IPL_DEPTH_8U, 1);

        register_to_cam< vision::Image<uint16_t, vision::DEPTH> >(camXd_, 100);
        imgXd_ = new vision::Image<uint16_t, vision::DEPTH>(camXd_->get_size());
        imgDispXd_ = new vision::Image<uint16_t, vision::DEPTH>(camXd_->get_size());

//        XnInt32 nMin;
//        XnInt32 nMax;
//        XnInt32 nStep;
//        int nStep;
//        XnInt32 nDefault;
//        XnBool autoSupported;

////        if(image_->IsCapabilitySupported(XN_CAPABILITY_LOW_LIGHT_COMPENSATION))
//        if(image_->IsCapabilitySupported(XN_CAPABILITY_HUE))
//        {
//            printf("Supplied image generator support LOW_LIGHT_COMPENSATION\n");
//            //return 1;
//        }
//        else
//        {
//            printf("Supplied image generator doesn't support LOW_LIGHT_COMPENSATION\n");
//        }
//
////for(int iCamIter = 0; iCamIter < 10; iCamIter++)
////{
////        image_->GetLowLightCompensationCap().GetRange(nMin, nMax, nStep, nDefault, autoSupported);
////
////        std::cout<<"nMin "<<(int)nMin<<" nMax "<<(int)nMax<<" nStep "<< (int)nStep <<" nDefault "<<(int)nDefault<<" autoSupported "<<(bool)autoSupported<<std::endl;
////
////sleep(1);
////} 
//
//        exit(-1);

        register_glfunc();

        finish_ = true;

        whiteboard_write<HandTrackerVSNI *>("plugin_handTrackerVSNI", this);
         
        return true ;
    }

    void HandTrackerVSNI::preloop_fct()
    {
        try
        {
            visionsystem::XMLRPCServer * server = whiteboard_read<visionsystem::XMLRPCServer *>("plugin_xmlrpc-server");
            if(server)
            {
                server->AddMethod(this);
            }
        }
        catch(...)
        {
            std::cout << "[handTrackerVSNI] No XML-RPC server plugin registered to the server, no biggies" << std::endl;
        }

        visionsystem::CameraDepthOpenNI* camXd = 0;
        camXd = dynamic_cast<visionsystem::CameraDepthOpenNI*>(camXd_);

        visionsystem::CameraImageOpenNI* camXi = 0;
        camXi = dynamic_cast<visionsystem::CameraImageOpenNI*>(camXi_);

        context_ = camXd->get_Context();
        depth_ = camXd->get_DepthGenerator();
        depthMD_ =  camXd->get_DepthMetaData();

        image_ = camXi->get_ImageGenerator();

        XnStatus rc = depth_->GetAlternativeViewPointCap().SetViewPoint(*image_); 

        if(rc != XN_STATUS_OK)
        {
             printf("Failed to set depth map mode \n");
        }
        else
        {
             printf("Succeed to set depth map mode \n");
        }

        initHandsGen();
        startHandsGen();
    }

    void HandTrackerVSNI::initHandsGen()
    {
        XnStatus nRetVal = XN_STATUS_OK;

        handsGen_.Create(*context_);

        nRetVal = handsGen_.Create(*context_);

        if(nRetVal != XN_STATUS_OK)
        {
             printf("Failed to create the hands generator \n");
        }
        else
        {
             printf("Succeed to create the hands generator \n");
        }

        nRetVal = handsGen_.RegisterHandCallbacks(Hand_Create, Hand_Update, Hand_Destroy, NULL, hHandCallbacks_);

        if(nRetVal != XN_STATUS_OK)
        {
             printf("Failed to register the callbacks of the hands generator \n");
        }
        else
        {
             printf("Succeed to register the callbacks of the hands generator \n");
        }

        printf("hands generetor created and initialized but not started\n");

   }

    void HandTrackerVSNI::startHandsGen()
    {
        handsGen_.StartGenerating();

        XnBool checkGenerate = handsGen_.IsGenerating();
        if(!checkGenerate)
        {
            std::cout<<"handsGenerator failed to start"<<std::endl;
        }
        else
        {
            std::cout<<"handsGenerator succeed to start"<<std::endl;
        }
    }

    void HandTrackerVSNI::stopHandsGen()
    {
    }

    void HandTrackerVSNI::Gesture_Recognized(  xn::GestureGenerator&   generator,
                                                            const XnChar*           strGesture,
                                                            const XnPoint3D*        pIDPosition,
                                                            const XnPoint3D*        pEndPosition,
                                                            void*                   pCookie)
    {
//        printf("Gesture recognized: %s\n", strGesture);
//    
//        HandTracker*    pThis = static_cast<HandTracker*>(pCookie);
////        if(sm_Instances.Find(pThis) == sm_Instances.end())
////        {
////            printf("Dead HandTracker: skipped!\n");
////            return;
////        }
//    
//        pThis->m_HandsGenerator.StartTracking(*pEndPosition);
    }

    void HandTrackerVSNI::Hand_Create(xn::HandsGenerator& generator, XnUserID nId, const XnPoint3D* pPosition, XnFloat fTime, void* pCookie)
    {
        printf("New Hand: %d @ (%f,%f,%f)\n", nId, pPosition->X, pPosition->Y, pPosition->Z);
    }

    void HandTrackerVSNI::Hand_Update( xn::HandsGenerator& generator, XnUserID nId, const XnPoint3D* pPosition, XnFloat fTime, void* pCookie)
    {
        printf("Hand Moving: %d @ (%f,%f,%f)\n", nId, pPosition->X, pPosition->Y, pPosition->Z);
    }

    void HandTrackerVSNI::Hand_Destroy(xn::HandsGenerator& generator, XnUserID nId, XnFloat fTime, void* pCookie)
    {
        printf("Lost Hand: %d\n", nId);
        //g_GestureGenerator.AddGesture(GESTURE_TO_USE, NULL);
    }

    void HandTrackerVSNI::loop_fct()
    {
        vision::Image<uint32_t, vision::RGB> * imgXi = this->dequeue_image< vision::Image<uint32_t, vision::RGB> > (camXi_);
        vision::Image<uint16_t, vision::DEPTH> * imgXd = this->dequeue_image< vision::Image<uint16_t, vision::DEPTH> > (camXd_);
//        if(finish_)
        {
//            finish_ = false;
//            t_.join();
            vision::Image<uint32_t, vision::RGB> * tmpXi = imgDispXi_;
            vision::Image<uint16_t, vision::DEPTH> * tmpXd = imgDispXd_;
            imgDispXi_ = imgXi_;
            imgDispXd_ = imgXd_;
            imgXi_ = tmpXi;
            imgXd_ = tmpXd;

            imgXi_->copy(imgXi);
            imgXd_->copy(imgXd);

//            t_ = boost::thread(boost::bind(&HandTrackerVSNI::calculs, this));
        }
        enqueue_image< vision::Image<uint32_t, vision::RGB> >(camXi_, imgXi);
        enqueue_image< vision::Image<uint16_t, vision::DEPTH> >(camXd_, imgXd);
    }

    void HandTrackerVSNI::calculs()
    {
        finish_ = true;
    }

    void HandTrackerVSNI::callback(visionsystem::Camera* cam, XEvent event)
    {
    }

    void HandTrackerVSNI::glfunc(visionsystem::Camera* cam)
    {
    }

    bool HandTrackerVSNI::post_fct()
    {
        std::cout << "[bci-self-interact-vision] post_fct()" << endl ;
        unregister_to_cam< vision::Image<uint32_t, vision::RGB> >( camXi_ );
        unregister_to_cam< vision::Image<uint16_t, vision::DEPTH> >( camXd_ );
        return true ;
    }

    void HandTrackerVSNI::execute(XmlRpcValue & params, XmlRpcValue & result)
    {
    }

    bool HandTrackerVSNI::GetObjectPositionNIHand(XmlRpc::XmlRpcValue & params, XmlRpc::XmlRpcValue & result)
    {
    }

} //end namespace handTrackerVSNI

PLUGIN(handTrackerVSNI::HandTrackerVSNI)
