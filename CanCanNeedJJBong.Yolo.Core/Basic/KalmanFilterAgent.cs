using System.Drawing;
using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core.Basic;

/// <summary>
/// 卡尔曼滤波代理
/// 用于二维平面物体运动的坐标预测,实现目标追踪的重要算法
/// </summary>
public class KalmanFilterAgent
{
    KalmanFilter kalman = new KalmanFilter(4, 2, 0);

    KalmanFilterAgent()
    {
        kalman.MeasurementMatrix = new Mat(2, 4, MatType.CV_32F, new float[]
        {
            1, 0, 0, 0,
            0, 1, 0, 0
        });
        
        //(状态转移矩阵）：该属性表示系统模型中状态变量的转移关系。状态转移矩阵描述了当前时刻状态向量与下一时刻状态向量之间的线性关系。它是一个矩阵，其维度为状态向量维度 x 状态向量维度。
        kalman.TransitionMatrix = new Mat(4, 4, MatType.CV_32F, new float[]
        {
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1
        });
        
        //（控制矩阵）：该属性表示外部控制输入对状态变量的影响关系。控制矩阵用于将外部控制输入与状态变量之间的关系建模。它是一个矩阵，其行数等于状态向量的维度，列数等于控制输入向量的维度。
        kalman.ControlMatrix = new Mat(4, 2, MatType.CV_32F, new float[]
        {
            0, 0,
            0, 0,
            1, 0,
            0, 1
        });
        
        // 过程噪声协方差矩阵）：该属性表示系统模型中过程噪声的协方差矩阵。它用于描述状态转移过程中的不确定性或噪声水平。过程噪声协方差矩阵通常也是一个对角矩阵，对角线上的元素表示各个状态变量的方差。
        kalman.ProcessNoiseCov = new Mat(4, 4, MatType.CV_32F, new float[]
        {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        });
        
        //(测量噪声协方差矩阵）：该属性表示观测噪声的协方差矩阵。它用于描述观测值的不确定性或噪声水平。测量噪声协方差矩阵通常是一个对角矩阵，对角线上的元素表示各个观测维度的方差。
        kalman.MeasurementNoiseCov = new Mat(2, 2, MatType.CV_32F, new float[]
        {
            1, 0,
            0, 1
        });
    }

    /// <summary>
    /// 预测下一个位置
    /// 二维平面预测下一个位置,通过几次预测和更新正确坐标,会得到较为准确的预测
    /// </summary>
    /// <returns>返回预测的坐标点</returns>
    public PointF NextLocation()
    {
        // 预测
        Mat Forecast = kalman.Predict();
        
        // 预测值
        PointF result = new PointF(Forecast.At<float>(0), Forecast.At<float>(1));
        
        return result;
    }

    /// <summary>
    /// 更新正确坐标
    /// 预测后通过更新正确的坐标,来逐步提高预测的准确性
    /// </summary>
    /// <param name="moodPoint">修正坐标</param>
    public void UpdatePoint(PointF moodPoint)
    {
        Mat up = new Mat(2, 1, MatType.CV_32F, new float[] { moodPoint.X, moodPoint.Y });
        
        kalman.Correct(up);
    }
}