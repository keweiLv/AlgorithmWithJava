package newStart;

import java.util.ArrayList;
import java.util.List;

public class Solution2 {

    // 两个非重叠子数组的最大和
    public int maxSumTwoNoOverlap(int[] nums, int firstLen, int secondLen) {
        int n = nums.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i++) {
            s[i + 1] = s[i] + nums[i];
        }
        return Math.max(f(s, firstLen, secondLen), f(s, secondLen, firstLen));
    }

    private int f(int[] s, int first, int sec) {
        int maxSumA = 0, res = 0;
        for (int i = first + sec; i < s.length; i++) {
            maxSumA = Math.max(maxSumA, s[i - sec] - s[i - sec - first]);
            res = Math.max(res, maxSumA + s[i] - s[i - sec]);
        }
        return res;
    }

    // 因子的组合
    public List<List<Integer>> getFactors(int n) {
        return dfs(2, n);
    }

    private List<List<Integer>> dfs(int start, int num) {
        if (num == 1) {
            return new ArrayList<>();
        }
        int qNum = (int) Math.sqrt(num);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = start; i <= qNum; i++) {
            if (num % i == 0) {
                List<Integer> simple = new ArrayList<>();
                simple.add(i);
                simple.add(num / i);
                result.add(simple);
                List<List<Integer>> nexList = dfs(i, num / i);
                for (List<Integer> list : nexList) {
                    list.add(i);
                    result.add(list);
                }
            }
        }
        return result;
    }

}
