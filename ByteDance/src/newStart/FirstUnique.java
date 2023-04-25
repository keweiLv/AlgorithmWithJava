package newStart;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * 第一个唯一数字
 */
public class FirstUnique {

    private HashSet<Integer> unique = new HashSet<>();
    private HashSet<Integer> repeated = new HashSet<>();
    private List<Integer> nums = new ArrayList<>();
    private int offset = 0;

    public FirstUnique(int[] nums) {
        for (int num : nums) {
            add(num);
        }
    }

    public int showFirstUnique() {
        if (unique.isEmpty()) {
            return -1;
        }
        for (int i = offset; i < nums.size(); i++) {
            if (unique.contains(nums.get(i))) {
                offset = i;
                return nums.get(i);
            }
        }
        return -1;
    }

    public void add(int value) {
        if (unique.contains(value)) {
            unique.remove(value);
            repeated.add(value);
        } else {
            if (!repeated.contains(value)) {
                unique.add(value);
                nums.add(value);
            }
        }
    }

    // 等差数列中缺失的数字
    public int missingNumber(int[] arr) {
        int sum = Arrays.stream(arr).sum();
        int n = arr.length;
        int or = (arr[0] + arr[n - 1]) * (n + 1) / 2;
        return or - sum;
    }

    // 按身高排序
    public String[] sortPeople(String[] names, int[] heights) {
        int n = names.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (i, j) -> heights[j] - heights[i]);
        String[] ans = new String[n];
        for (int i = 0; i < n; i++) {
            ans[i] = names[idx[i]];
        }
        return ans;
    }

    // 形成字符串的最短路径
    public int shortestWay(String source, String target) {
        int n = source.length();
        int j = 0;
        int ans = 0;
        while (j < target.length()) {
            int tmp = j;
            for (int i = 0; i < n; i++) {
                if (j < target.length() && source.charAt(i) == target.charAt(j)) {
                    j++;
                }
            }
            if (tmp == j) {
                return -1;
            }
            ans++;
        }
        return ans;
    }

    // 粉刷房子
    public int minCost(int[][] costs) {
        int redCost = costs[0][0], blueCost = costs[0][1], greenCost = costs[0][2];
        for (int i = 1; i < costs.length; i++) {
            int newR = Math.min(blueCost, greenCost) + costs[i][0];
            int newB = Math.min(redCost, greenCost) + costs[i][1];
            int newG = Math.min(redCost, blueCost) + costs[i][2];
            redCost = newR;
            blueCost = newB;
            greenCost = newG;
        }
        return Math.min(redCost, Math.min(blueCost, greenCost));
    }

    // 统计只含单一字母的子串
    public int countLetters(String s) {
        int ans = 0;
        int count = 1;
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == s.charAt(i - 1)) {
                count++;
            } else {
                ans += count * (count + 1) / 2;
                count = 1;
            }
        }
        ans += count * (count + 1) / 2;
        return ans;
    }


}
