package LC_RACE.Biweekly_108;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 第 108 场双周赛
 */
public class Solution {

    // 最长交替子序列
    public static int alternatingSubarray(int[] nums) {
        int ans = 0;
        int l = 0, r = 1;
        int flag = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] - nums[i - 1] == flag) {
                r++;
                flag *= -1;
                ans = Math.max(ans, r - l);
            } else {
                l += 1;
                r = l + 1;
                i = l;
                flag = 1;
            }
        }
        return ans == 0 ? -1 : ans;
    }


    // 重新放置石块
    public List<Integer> relocateMarbles(int[] nums, int[] moveFrom, int[] moveTo) {
        Set<Integer> record = new HashSet<>();
        for (int num : nums) {
            record.add(num);
        }
        int n = moveFrom.length;
        for (int i = 0; i < n; i++) {
            Integer from = moveFrom[i];
            Integer to = moveTo[i];
            record.remove(from);
            record.add(to);
        }
        return record.stream().sorted().collect(Collectors.toList());
    }


    public static void main(String[] args) {
        int i = minimumBeautifulSubstrings("1011");
        System.out.println(i);
    }

    // 将字符串分割为最少的美丽子字符串
    public static int minimumBeautifulSubstrings(String s) {
        int n = s.length();
        Set<String> pows = new HashSet<>();
        for (int i = 0; i < 22; i++) {
            pows.add(Long.toBinaryString((long) Math.pow(5, i)));
        }
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = i; j > 0 && i - j <= 20; j--) {
                if (s.charAt(j - 1) == '0') {
                    continue;
                }
                if (pows.contains(s.substring(j - 1, i))) {
                    dp[i] = Math.min(dp[i], dp[j - 1] != Integer.MAX_VALUE ? dp[j - 1] + 1 : Integer.MAX_VALUE);
                }
            }
        }
        return dp[n] == Integer.MAX_VALUE ? -1 : dp[n];
    }

}
