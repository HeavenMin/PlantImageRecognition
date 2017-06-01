

#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

#include <memory>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/memmapped_file_system.h"

@interface CameraExampleViewController
: UIViewController<UIGestureRecognizerDelegate,
AVCaptureVideoDataOutputSampleBufferDelegate> {
    IBOutlet UIView *previewView;
    IBOutlet UISegmentedControl *camerasControl;
    AVCaptureVideoPreviewLayer *previewLayer;
    AVCaptureVideoDataOutput *videoDataOutput;
    dispatch_queue_t videoDataOutputQueue;
    AVCaptureStillImageOutput *stillImageOutput;
    UIView *flashView;
    BOOL isUsingFrontFacingCamera;
    AVSpeechSynthesizer *synth;
    NSMutableDictionary *oldPredictionValues;
    NSMutableArray *labelLayers;
    AVCaptureSession *session;
    std::unique_ptr<tensorflow::Session> tf_session;
    std::unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;
    std::vector<std::string> labels;
}
@property(strong, nonatomic) CATextLayer *predictionTextLayer;

- (IBAction)takePicture:(id)sender;
- (IBAction)switchCameras:(id)sender;
- (IBAction)getPhotoLib:(id)sender;
- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section;
- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath;

- (IBAction)flowerSet:(id)sender;
- (IBAction)backToCamera:(id)sender;

@end
