package start240304;

/**
 * 区域和检索-数组不可变
 */
public class NumArray {

    int[] nums;

    public NumArray(int[] _nums) {
        nums = new int[_nums.length + 1];
        for (int i = 1; i <= _nums.length; i++) {
            nums[i] = _nums[i - 1] + nums[i - 1];
        }
    }

    public int sumRange(int left, int right) {
        return nums[right + 1] - nums[left];
    }


}
