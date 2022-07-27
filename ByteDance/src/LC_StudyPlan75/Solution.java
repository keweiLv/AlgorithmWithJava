package LC_StudyPlan75;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

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
			if (2 * sum + nums[i] == totel) {
				return i;
			}
			sum += nums[i];
		}
		return -1;
	}

	// 同构字符串
	public boolean isIsomorphic(String s, String t) {
		Map<Character, Character> s2t = new HashMap<>(), t2s = new HashMap<>();
		for (int i = 0; i < s.length(); i++) {
			char a = s.charAt(i), b = t.charAt(i);
			if (s2t.containsKey(a) && s2t.get(a) != b || t2s.containsKey(b) && t2s.get(b) != a) {
				return false;
			}
			s2t.put(a, b);
			t2s.put(b, a);
		}
		return true;
	}

	// 判断子序列
	public boolean isSubsequence(String s, String t) {
		int n = s.length(), m = t.length();
		int i = 0, j = 0;
		while (i < n && j < m) {
			if (s.charAt(i) == t.charAt(j)) {
				i++;
			}
			j++;
		}
		return i == n;
	}
}
