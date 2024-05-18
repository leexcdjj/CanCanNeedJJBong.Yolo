using CanCanNeedJJBong.Yolo.Core.Basic;

namespace CanCanNeedJJBong.Yolo.Core;

public static class YoloHelper
{
    /// <summary>
    /// 置信度排序
    /// </summary>
    /// <param name="data"></param>
    public static void RankConfidenceDegree(List<YoloData> data)
    {
        if (data == null || data.Count == 0)
        {
            return;
        }

        if (data[0].BasicData.Length == 2)
        {
            // 使用内置的排序算法，根据 BasicData[0] 降序排序
            data.Sort((x, y) => y.BasicData[0].CompareTo(x.BasicData[0]));
        }
        else
        {
            // 使用内置的排序算法，根据 BasicData[4] 降序排序
            data.Sort((x, y) => y.BasicData[4].CompareTo(x.BasicData[4]));
        }
    }
    
    /// <summary>
    /// nms过滤
    /// </summary>
    /// <param name="filterDataList">首次过滤数组</param>
    /// <param name="iouThreshold">iou阈值</param>
    /// <param name="allIou">全局iou</param>
    /// <returns>经过nms过滤后的Yolo数据列表</returns>
    public static List<YoloData> NMSFilter(List<YoloData> filterDataList, float iouThreshold, bool allIou)
    {
        // 对数据按置信度降序排序
        YoloHelper.RankConfidenceDegree(filterDataList);

        // 保存过滤后的结果
        List<YoloData> result = new List<YoloData>();

        foreach (var data in filterDataList)
        {
            bool isFilter = true;

            foreach (var res in result)
            {
                // 判断是否计算全局iou或属于同一类
                if (allIou || data.BasicData[5] == res.BasicData[5])
                {
                    // 计算两个矩形的交并比
                    float iou = CalcCrossMergeRatio(data.BasicData, res.BasicData);

                    // 如果iou超过阈值，设置isFilter为false
                    if (iou > iouThreshold)
                    {
                        isFilter = false;
                        break;
                    }
                }
            }

            // 如果通过过滤，将数据添加到结果中
            if (isFilter) result.Add(data);
        }

        return result;
    }
    
    #region 计算交并比 CalcCrossMergeRatio

    /// <summary>
    /// 计算交并比（IoU）
    /// </summary>
    /// <param name="rectangle1">矩形1</param>
    /// <param name="rectangle2">矩形2</param>
    /// <returns>交并比（IoU）</returns>
    private static float CalcCrossMergeRatio(float[] rectangle1, float[] rectangle2)
    {
        // 计算矩形1的边界
        float[] rect1Bounds = CalculateBounds(rectangle1);
        // 计算矩形2的边界
        float[] rect2Bounds = CalculateBounds(rectangle2);

        // 计算交集区域
        float intersectionArea = CalculateIntersectionArea(rect1Bounds, rect2Bounds);

        // 计算矩形1和矩形2的面积
        float area1 = CalculateArea(rect1Bounds);
        float area2 = CalculateArea(rect2Bounds);

        // 计算并集区域
        float unionArea = area1 + area2 - intersectionArea;

        // 返回交并比（IoU）
        return intersectionArea / unionArea;
    }

    /// <summary>
    /// 计算矩形的边界
    /// </summary>
    /// <param name="rectangle">矩形</param>
    /// <returns>矩形的边界数组 [left, top, right, bottom]</returns>
    private static float[] CalculateBounds(float[] rectangle)
    {
        return new float[]
        {
            rectangle[0] - rectangle[2] / 2, // left
            rectangle[1] - rectangle[3] / 2, // top
            rectangle[0] + rectangle[2] / 2, // right
            rectangle[1] + rectangle[3] / 2 // bottom
        };
    }

    /// <summary>
    /// 计算两个矩形的交集面积
    /// </summary>
    /// <param name="rect1">矩形1的边界</param>
    /// <param name="rect2">矩形2的边界</param>
    /// <returns>交集面积</returns>
    private static float CalculateIntersectionArea(float[] rect1, float[] rect2)
    {
        float leftBoundary = Math.Max(rect1[0], rect2[0]);
        float topBoundary = Math.Max(rect1[1], rect2[1]);
        float rightBoundary = Math.Min(rect1[2], rect2[2]);
        float bottomBoundary = Math.Min(rect1[3], rect2[3]);

        if (leftBoundary < rightBoundary && topBoundary < bottomBoundary)
        {
            return (rightBoundary - leftBoundary) * (bottomBoundary - topBoundary);
        }

        return 0;
    }

    /// <summary>
    /// 计算矩形的面积
    /// </summary>
    /// <param name="bounds">矩形的边界</param>
    /// <returns>矩形的面积</returns>
    private static float CalculateArea(float[] bounds)
    {
        return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
    }

    #endregion
}