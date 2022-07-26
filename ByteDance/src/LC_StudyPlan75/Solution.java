package LC_StudyPlan75;

import java.util.Arrays;

/**
 * @author Kezi
 * @date 2022年07月26日 22:32
 */
public class Solution {

	// 一维数组的动态和
	public int[] runningSum(int[] nums) {
		for (int i = 1; i < nums.length; i++) {
			nums[i] = nums[i - 1] + nums[i];
		}
		return nums;
	}

	// 寻找数组的中心索引
	public int pivotIndex(int[] nums) {
		int totel = Arrays.stream(nums).sum();
		int sum = 0;
		for (int i = 0; i < nums.length; i++) {
			if (2 * sum + nums[i] == totel){
				return i;
			}
			sum += nums[i];
		}
		return -1;
	}
}
