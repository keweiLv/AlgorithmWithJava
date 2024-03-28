package start240304;

import java.util.*;

public class Solution {

    // 最短单词距离
    public int shortestDistance(String[] wordsDict, String word1, String word2) {
        int len = wordsDict.length;
        int ans = len;
        int idx1 = -1, idx2 = -1;
        for (int i = 0; i < len; i++) {
            String word = wordsDict[i];
            if (word.equals(word1)) {
                idx1 = i;
            } else if (word.equals(word2)) {
                idx2 = i;
            }
            if (idx1 >= 0 && idx2 >= 0) {
                ans = Math.min(ans, Math.abs(idx1 - idx2));
            }
        }
        return ans;
    }

    // 三角形最小路径和
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] dp = new int[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }

    // 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int max = 0, left = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }

    // 找出美丽数组的最小和
    final int MOD = (int) 1e9 + 7;

    public int minimumPossibleSum(int n, int target) {
        int m = target / 2;
        if (n <= m) {
            return (int) ((long) (1 + n) * n / 2 % MOD);
        }
        return (int) (((long) (1 + m) * m / 2 + ((long) target + target + (n - m) - 1) * (n - m) / 2) % MOD);
    }

    // 猜数字游戏
    public String getHint(String secret, String guess) {
        int bulls = 0;
        int[] cntS = new int[10];
        int[] cntG = new int[10];
        for (int i = 0; i < guess.length(); i++) {
            if (secret.charAt(i) == guess.charAt(i)) {
                bulls++;
            } else {
                cntS[secret.charAt(i) - '0']++;
                cntG[guess.charAt(i) - '0']++;
            }
        }
        int cows = 0;
        for (int i = 0; i < 10; i++) {
            cows += Math.min(cntS[i], cntG[i]);
        }
        return Integer.toString(bulls) + "A" + Integer.toString(cows) + "B";
    }

    // 二叉树的层平均值
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> averages = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            double sum = 0;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode poll = queue.poll();
                sum += poll.val;
                TreeNode left = poll.left;
                TreeNode right = poll.right;
                if (left != null) {
                    queue.offer(left);
                }
                if (right != null) {
                    queue.offer(right);
                }
            }
            averages.add(sum / size);
        }
        return averages;
    }

    // 最大二进制奇数
    public String maximumOddBinaryNumber(String s) {
        int cnt = (int) s.chars().filter(c -> c == '1').count();
        return "1".repeat(cnt - 1) + "0".repeat(s.length() - cnt) + "1";
    }

    // 合并后数组中的最大元素
    public long maxArrayValue(int[] nums) {
        int n = nums.length;
        long sum = nums[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            sum = nums[i] <= sum ? sum + nums[i] : nums[i];
        }
        return sum;
    }

    // 好子数组的最大分数
    public int maximumScore(int[] nums, int k) {
        int n = nums.length;
        int ans = nums[k];
        int min = nums[k];
        int left = k;
        int right = k;
        while (left > 0 || right + 1 < n) {
            if (right + 1 >= n || (left > 0 && nums[left - 1] > nums[right + 1])) {
                left--;
                min = Math.min(min, nums[left]);
            } else {
                right++;
                min = Math.min(min, nums[right]);
            }
            ans = Math.max(ans, (right - left + 1) * min);
        }
        return ans;
    }

    // 柱装图中最大的矩形
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            int x = heights[i];
            while (!stack.isEmpty() && x <= heights[stack.peek()]) {
                stack.pop();
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        int[] right = new int[n];
        stack.clear();
        for (int i = n - 1; i >= 0; i--) {
            int x = heights[i];
            while (!stack.isEmpty() && x <= heights[stack.peek()]) {
                stack.pop();
            }
            right[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans = Math.max(ans, heights[i] * (right[i] - left[i] - 1));
        }
        return ans;
    }

    // 零钱兑换二
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }

    // 二叉树的右视图
    List<Integer> res = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        dfs(root, 0);
        return res;
    }

    private void dfs(TreeNode root, int dep) {
        if (root == null) {
            return;
        }
        if (dep == res.size()) {
            res.add(root.val);
        }
        dfs(root.right, dep + 1);
        dfs(root.left, dep + 1);
    }

    // 统计将重叠区间合并的方案数
    public int countWays(int[][] ranges) {
        Arrays.sort(ranges, (a, b) -> a[0] - b[0]);
        int ans = 1;
        int maxR = -1;
        for (int[] p : ranges) {
            if (p[0] > maxR) {
                ans = ans * 2 % 1000000007;
            }
            maxR = Math.max(maxR, p[1]);
        }
        return ans;
    }

    // 访问完所有房间的第一天
    public int firstDayBeenInAllRooms(int[] nextVisit) {
        final long MOD = 1000000007;
        int n = nextVisit.length;
        long[] s = new long[n];
        for (int i = 0; i < n - 1; i++) {
            int j = nextVisit[i];
            s[i + 1] = (s[i] * 2 - s[j] + 2 + MOD) % MOD;
        }
        return (int) s[n - 1];
    }
}

