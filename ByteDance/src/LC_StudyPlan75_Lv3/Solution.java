package LC_StudyPlan75_Lv3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Kezi
 * @date 2022年09月08日 0:02
 */
public class Solution {

	/**
	 * 位1的个数
	 * 算术右移 >> ：舍弃最低位，高位用符号位填补；
	 * 逻辑右移 >>> ：舍弃最低位，高位用 0 填补
	 * 对于负数而言，其二进制最高位是 1，如果使用算术右移，那么高位填补的仍然是 1。也就是 n 永远不会为 0
	 */
	public int hammingWeight(int n) {
		int count = 0;
		while (n != 0) {
			count += n & 1;
			n >>>= 1;
		}
		return count;
	}

	// 只出现一次的数字
	public int singleNumber(int[] nums) {
		int ans = nums[0];
		for (int i = 1; i < nums.length; i++) {
			ans = ans ^ nums[i];
		}
		return ans;
	}

	// 子集Ⅱ
	private List<List<Integer>> ans;
	private List<Integer> path;

	public List<List<Integer>> subsetsWithDup(int[] nums) {
		ans = new ArrayList<>();
		path = new ArrayList<>();
		Arrays.sort(nums);
		int n = nums.length;
		boolean[] visited = new boolean[n];
		// 开始回溯
		backtrace(nums, 0, visited, n);
		return ans;
	}

	private void backtrace(int[] nums, int start, boolean[] visited, int n) {
		ans.add(new ArrayList<>(path));
		for (int i = start; i < n; i++) {
			if (i > 0 && nums[i - 1] == nums[i] && !visited[i - 1]) {
				continue;
			}
			visited[i] = true;
			path.add(nums[i]);
			backtrace(nums, i + 1, visited, n);
			visited[i] = false;
			path.remove(path.size() - 1);
		}
	}

	// 字母异位词分组
	public List<List<String>> groupAnagrams(String[] strs) {
		return new ArrayList<>(Arrays.stream(strs)
				.collect(Collectors.groupingBy(str -> {
					char[] chars = str.toCharArray();
					Arrays.sort(chars);
					return new String(chars);
				})).values()
		);
	}
}
