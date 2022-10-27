package AlgInterviewSummary;

/**
 * @author Kezi
 * @date 2022年10月27日 23:01
 * @describe leetcode算法面试题汇总
 */
public class Solution {

	// 搜索二维矩阵Ⅱ
	public boolean searchMatrix(int[][] matrix, int target) {
		int row = matrix.length - 1, col = 0;
		while (row >= 0 && col < matrix[0].length) {
			if (target > matrix[row][col]) {
				col++;
			} else if (target < matrix[row][col]) {
				row--;
			} else {
				return true;
			}
		}
		return false;
	}

	// 合并两个有序数组
	public void merge(int[] nums1, int m, int[] nums2, int n) {
		int i = m - 1;
		int j = n - 1;
		int end = m + n - 1;
		while (j >= 0) {
			nums1[end--] = (i >= 0 && nums1[i] > nums2[j])?nums1[i--]:nums2[j--];
		}
	}
}
