package LC_RACE.Biweekly_105;

import java.util.*;

/**
 * 第105场双周赛
 */
public class Solution {

    // 购买两块巧克力
    public int buyChoco(int[] prices, int money) {
        int res = 0;
        Arrays.sort(prices);
        return money - (prices[0] + prices[1]) >= 0 ? money - (prices[0] + prices[1]) : money;
    }

    // 字符串中的额外字符
    public int minExtraChar(String s, String[] dictionary) {
        Set<String> set = new HashSet<>(Arrays.asList(dictionary));
        int n = s.length();
        int[] f = new int[n + 1];
        for (int i = 0; i < n; i++) {
            f[i + 1] = f[i] + 1;
            for (int j = 0; j < i + 1; j++) {
               if (set.contains(s.substring(j,i+1))){
                   f[i+1] = Math.min(f[i+1],f[j]);
               }
            }
        }
        return f[n];
    }

    // 一个小组的最大实力值
    public long maxStrength(int[] nums) {
        if (nums.length <= 1 && nums[0] < 0) {
            return 0;
        }
        long res = 0;
        int cnt = 0;
        int mark = -1;
        boolean flag = false;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] >= 0) {
                break;
            } else {
                cnt++;
                mark = i;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                continue;
            }
            if ((res == 0 && i != mark) || cnt % 2 == 0) {
                if (!flag) {
                    res = 1;
                    flag = true;
                }
            }
            if (cnt % 2 == 0) {
                res *= nums[i];
            } else {
                if (i != mark) {
                    res *= nums[i];
                }
            }
        }
        return res;
    }

}
