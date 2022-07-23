package raceCollection;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Kezi
 * @date 2022年07月23日 22:47
 * 禾赛科技,第 83 场双周赛
 */
public class BiweeklyGame_83rd {

	// 最好的扑克手牌
	public static String bestHand(int[] ranks, char[] suits) {
		boolean flush = true, three = true, pair = true, high = true;
		Map<Integer, Integer> countMap = new HashMap<>(16);
		for (int i = 0; i < ranks.length - 1; i++) {
			if (suits[i] != suits[i + 1]) {
				flush = false;
			}
			countMap.put(ranks[i], countMap.getOrDefault(ranks[i], 0) + 1);
			if (i == ranks.length - 2) {
				countMap.put(ranks[i + 1], countMap.getOrDefault(ranks[i + 1], 0) + 1);
			}
		}
		if (flush) {
			return "Flush";
		}
		if (countMap.size() == 5) {
			return "High Card";
		}
		int max = 0;
		for (Integer key : countMap.keySet()) {
			if (countMap.get(key) > max) {
				max = countMap.get(key);
			}
		}
		if (max >= 3) {
			return "Three of a Kind";
		}
		return "Pair";
	}

	// 全0子数组的数目
	// 结果设为long类型，防止溢出
	public static long zeroFilledSubarray(int[] nums) {
		long count = 0;
		int len = 0;
		for (int fast = 0; fast < nums.length; fast++) {
			if (nums[fast] == 0) {
				len++;
			}
			if (nums[fast] != 0 || fast == nums.length - 1) {
				for (int i = 1; i <= len; i++) {
					count += i;
				}
				len = 0;
			}
		}
		return count;
	}
}
