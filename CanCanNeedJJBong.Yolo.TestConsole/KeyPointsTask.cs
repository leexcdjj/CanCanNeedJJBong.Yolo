namespace CanCanNeedJJBong.Yolo.TestConsole;

class KeyPointsTask : YoloTask
{
    protected override (string testImgName, string modelName, string action, int taskMode) GetTaskDetails()
    {
        return ("4.bmp", "yolov8n-pose.transd.onnx", "KeyPoints", 5);
    }
}