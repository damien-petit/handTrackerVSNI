#ifndef HAND_TRACKER_VSNI
#define HAND_TRACKER_VSNI

#include <configparser/configparser.h>
#include <vision/vision.h>
#include <visionsystem/plugin.h>
#include <visionsystem/viewer.h>

#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>

#include "XmlRpc.h"

//CoshellClient is used to obtain the head position
#include <coshell-client/CoshellClient.h>

//openni

#include <XnOS.h>
//#include <XnCppWrapper.h>
#include <XnOpenNI.h>
#include <XnCodecIDs.h>
#include <XnCppWrapper.h>
#include <XnPropNames.h>

#include <opencv2/opencv.hpp>

#include "NiTrailHistory.h"

namespace handTrackerVSNI
{

    class HandTrackerVSNI : public visionsystem::Plugin, public visionsystem::WithViewer, public configparser::WithConfigFile, public XmlRpc::XmlRpcServerMethod {

        public:

            HandTrackerVSNI( visionsystem::VisionSystem *vs, string sandbox ) ;
            ~HandTrackerVSNI() ;

            bool pre_fct()  ;
            void preloop_fct()  ;
            void loop_fct() ;
            bool post_fct() ;
            void calculs();
            /*virtual function of configparser used by  read_config_file of configparser::WithConfigFile*/
            void parse_config_line( vector<string> &line );
            void callback(visionsystem::Camera* cam, XEvent event);
            void glfunc(visionsystem::Camera* cam);

            /* Allow XML-RPC remote call */
            /*
                params should be a string with object name
                result will be this way: { "left" : { x, y }, "right" : { x, y } }
            */
            void execute(XmlRpc::XmlRpcValue & params, XmlRpc::XmlRpcValue & result);

            bool GetObjectPositionNIHand(XmlRpc::XmlRpcValue & params, XmlRpc::XmlRpcValue & result);
            bool GetObjectPositionNIHand2(XmlRpc::XmlRpcValue & params, XmlRpc::XmlRpcValue & result);

            void initData();
            
            void initHandsGen();
            void startHandsGen();
            void stopHandsGen();

//OpenNI callback
            static void XN_CALLBACK_TYPE Gesture_Recognized(xn::GestureGenerator&   generator,
                                                    const XnChar*           strGesture,
                                                    const XnPoint3D*        pIDPosition,
                                                    const XnPoint3D*        pEndPosition,
                                                    void*                   pCookie);
            static void XN_CALLBACK_TYPE Gesture_Process(   xn::GestureGenerator&   generator,
                                                    const XnChar*           strGesture,
                                                    const XnPoint3D*        pPosition,
                                                    XnFloat                 fProgress,
                                                    void*                   pCookie)    {}
            static void XN_CALLBACK_TYPE Hand_Create(xn::HandsGenerator& generator, XnUserID nId, const XnPoint3D* pPosition, XnFloat fTime, void* pCookie);
            static void XN_CALLBACK_TYPE Hand_Update(xn::HandsGenerator& generator, XnUserID nId, const XnPoint3D* pPosition, XnFloat fTime, void* pCookie);
            static void XN_CALLBACK_TYPE Hand_Destroy(xn::HandsGenerator& generator, XnUserID nId, XnFloat fTime, void* pCookie);

        private:

            std::string sandbox_;

            vision::Image<uint32_t, vision::RGB> * imgXi_;
            vision::Image<uint8_t, vision::MONO> * imgXiMONO_;
            vision::Image<uint32_t, vision::RGB> * imgDispXi_;
            vision::Image<uint16_t, vision::DEPTH> * imgXd_;
            vision::Image<uint16_t, vision::DEPTH> * imgDispXd_;

            bool finish_;

            visionsystem::Camera * camXi_;
            visionsystem::Camera * camXd_;

//TODO thread used for calcul it may be unnecessary
            boost::thread t_;

            std::string camNameXi_;
            std::string camNameXd_;

            cv::Mat cameraMatrixXtionRGB_;
            cv::Mat distorsionMatrixXtionRGB_;
            cv::Mat headToCamMatrixXtionRGB_;

            cv::Mat NIHandMatMono_;

// used by the openni hand tracker function
            xn::Context* context_;
            xn::DepthGenerator* depth_;
            xn::ImageGenerator* image_;
            xn::DepthMetaData* depthMD_;

            XnCallbackHandle hHandCallbacks_;
            xn::HandsGenerator handsGen_;
            xn::GestureGenerator gestureGen_;

            TrailHistory handsHistory_;
// end used by the openni hand tracker function

            

            std::string coshellName_;
            coshell::CoshellClient * coshellVision_;

    };
} // namespace handTrackerVSNI

#endif
