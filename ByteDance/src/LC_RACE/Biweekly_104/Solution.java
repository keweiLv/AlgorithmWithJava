package LC_RACE.Biweekly_104;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 第104场双周赛
 */
public class Solution {

    // 老人的数目
    public int countSeniors(String[] details) {
        int ans = 0;
        for (String p : details) {
            String substring = p.substring(11, 13);
            if (Integer.parseInt(substring) > 60) {
                ans++;
            }
        }
        return ans;
    }

    // 矩阵中的和
    public static int matrixSum(int[][] nums) {
        int ans = 0;
        boolean flag = true;
        int cnt = 0;
        while (flag) {
            int max = 0;
            out:
            for (int[] row : nums) {
                int curMax = Arrays.stream(row).max().getAsInt();
                for (int i = 0; i < row.length; i++) {
                    if (row[i] == curMax) {
                        max = Math.max(max, curMax);
                        row[i] = 0;
                        continue out;
                    }
                }
            }
            ans += max;
            if (cnt++ == nums[0].length) {
                flag = false;
            }
        }
        return ans;
    }


    /**
     * 英雄的力量
     * TODO 超时
      */
    static List<Integer> path = new ArrayList<>();
    static int[] nums;
    static int ans = 0;

    public static int sumOfPower(int[] _nums) {
        nums = _nums;
        dfs(0);
        return ans;
    }

    private static void dfs(int i) {
        if (i == nums.length) {
            List<Integer> t = new ArrayList<>(path);
            if (t.size() > 0){
                int[] tmp = new int[t.size()];
                for (int j = 0; j < t.size(); j++) {
                    tmp[j] = t.get(j);
                }
                int max = Arrays.stream(tmp).max().getAsInt();
                int min = Arrays.stream(tmp).min().getAsInt();
                int sum = max * max * min;
                ans = (int) ((ans + sum) % (1e9+7));
            }
            return;
        }
        dfs(i + 1);
        path.add(nums[i]);
        dfs(i + 1);
        path.remove(path.size() - 1);
    }

    public static void main(String[] args) {
        int[] param = new int[]{658,489,777,2418,1893,130,2448,178,1128,2149,1059,1495,1166,608,2006,713,1906,2108,680,1348,860,1620,146,2447,1895,1083,1465,2351,1359,1187,906,533,1943,1814,1808,2065,1744,254,1988,1889,1206};

        System.out.println(sumOfPower(param));
    }
}
