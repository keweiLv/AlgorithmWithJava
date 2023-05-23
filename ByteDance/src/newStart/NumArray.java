package newStart;

/**
 * 区域和检索-数组不可变
 */
public class NumArray {

    int[] sum;

    public NumArray(int[] nums) {
        int n = nums.length;
        sum = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i - 1] + nums[i - 1];
        }
    }

    public int sumRange(int left, int right) {
        left++;
        right++;
        return sum[right] - sum[left - 1];
    }
}
