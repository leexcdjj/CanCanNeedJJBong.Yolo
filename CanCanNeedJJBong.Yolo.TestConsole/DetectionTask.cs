namespace CanCanNeedJJBong.Yolo.TestConsole;

class DetectionTask : YoloTask
{
    protected override (string testImgName, string modelName, string action, int taskMode) GetTaskDetails()
    {
        return ("OBB测试飞机场2.bmp", "yolov9-c.onnx", "Detection", 1);
    }
}