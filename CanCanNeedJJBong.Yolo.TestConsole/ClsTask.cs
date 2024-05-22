namespace CanCanNeedJJBong.Yolo.TestConsole;

class ClsTask : YoloTask
{
    protected override (string testImgName, string modelName, string action, int taskMode) GetTaskDetails()
    {
        return ("螺丝刀2.jpg", "yolov8n-cls.onnx", "cls", 0);
    }
}