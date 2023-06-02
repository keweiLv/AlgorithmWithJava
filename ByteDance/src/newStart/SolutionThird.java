package newStart;

import java.util.*;

public class SolutionThird {

    // 可被三整除的偶数的平均值
    public int averageValue(int[] nums) {
        int s = 0, n = 0;
        for (int num : nums) {
            if (num % 3 == 0 && num % 2 == 0) {
                s += num;
                n++;
            }
        }
        return n == 0 ? 0 : s / n;
    }

    // 二叉树最大宽度
    Map<Integer, Integer> map = new HashMap<>();
    int ans;

    public int widthOfBinaryTree(TreeNode root) {
        dfs(root, 1, 0);
        return ans;
    }

    private void dfs(TreeNode root, int v, int depth) {
        if (root == null) {
            return;
        }
        if (!map.containsKey(depth)) {
            map.put(depth, v);
        }
        ans = Math.max(ans, v - map.get(depth) + 1);
        v = v - map.get(depth) + 1;
        dfs(root.left, v << 1, depth + 1);
        dfs(root.right, v << 1 | 1, depth + 1);
    }

    // 删点成林
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        List<TreeNode> ans = new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        for (int i : to_delete) {
            set.add(i);
        }
        if (dfs(root, ans, set) != null) {
            ans.add(root);
        }
        return ans;
    }

    private TreeNode dfs(TreeNode root, List<TreeNode> ans, Set<Integer> set) {
        if (root == null) {
            return null;
        }
        root.left = dfs(root.left, ans, set);
        root.right = dfs(root.right, ans, set);
        if (!set.contains(root.val)) {
            return root;
        }
        if (root.left != null) {
            ans.add(root.left);
        }
        if (root.right != null) {
            ans.add(root.right);
        }
        return null;
    }

    // 数组中两个数的最大异或值
    public int findMaximumXOR(int[] nums) {
        int res = 0;
        int mask = 0;
        for (int i = 30; i >= 0; i--) {
            mask = mask | (1 << i);
            Set<Integer> set = new HashSet<>();
            for (int num : nums) {
                set.add(num & mask);
            }
            int temp = res | (1 << i);
            for (Integer pre : set) {
                if (set.contains(temp ^ pre)) {
                    res = temp;
                    break;
                }
            }
        }
        return res;
    }

    // 使用最小花费爬楼梯
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
    }

    // 叶值的最小代价生成树
    public int mctFromLeafValues(int[] arr) {
        Deque<Integer> stack = new ArrayDeque<>();
        stack.offerLast(Integer.MAX_VALUE);
        int ans = 0;
        for (int i = 0; i < arr.length; i++) {
            while (arr[i] > stack.peekLast()) {
                ans += stack.pollLast() * Math.min(stack.peekLast(), arr[i]);
            }
            stack.offerLast(arr[i]);
        }
        while (stack.size() > 2) {
            ans += stack.pollLast() * stack.peekLast();
        }
        return ans;
    }

    // 礼盒的最大甜蜜度
    public int maximumTastiness(int[] price, int k) {
        Arrays.sort(price);
        int left = 0, right = (price[price.length - 1] - price[0]) / (k - 1) + 1;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (check(price, mid) >= k) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return left;
    }

    private int check(int[] price, int d) {
        int cnt = 1, pre = price[0];
        for (int p : price) {
            if (p >= pre + d) {
                cnt++;
                pre = p;
            }
        }
        return cnt;
    }

    // 爱吃香蕉的珂珂
    public int minEatingSpeed(int[] piles, int h) {
        int l = 0, r = (int) 1e9;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (check(piles, mid, h)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    private boolean check(int[] piles, int mid, int h) {
        int ans = 0;
        for (int p : piles) {
            ans += Math.ceil(p * 1.0 / mid);
        }
        return ans <= h;
    }

    // 统计范围内的元音字符串数
    public int[] vowelStrings(String[] words, int[][] queries) {
        Set<Character> set = Set.of('a', 'e', 'i', 'o', 'u');
        int n = words.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i++) {
            char a = words[i].charAt(0), b = words[i].charAt(words[i].length() - 1);
            s[i + 1] = s[i] + (set.contains(a) && set.contains(b) ? 1 : 0);
        }
        int m = queries.length;
        int[] ans = new int[m];
        for (int i = 0; i < m; i++) {
            int l = queries[i][0], r = queries[i][1];
            ans[i] = s[r + 1] - s[l];
        }
        return ans;
    }

    // 删除并获得点数
    int[] cnts = new int[10010];

    public int deleteAndEarn(int[] nums) {
        int n = nums.length;
        int max = 0;
        for (int nu : nums) {
            cnts[nu]++;
            max = Math.max(max, nu);
        }
        int[][] f = new int[max + 1][2];
        for (int i = 1; i <= max; i++) {
            f[i][0] = Math.max(f[i - 1][0], f[i - 1][1]);
            f[i][1] = f[i - 1][0] + i * cnts[i];
        }
        return Math.max(f[max][0], f[max][1]);
    }

    // 最长湍流子数组
    public int maxTurbulenceSize(int[] arr) {
        int n = arr.length, ans = 1;
        int[][] f = new int[n][2];
        f[0][0] = f[0][1] = 1;
        for (int i = 1; i < n; i++) {
            f[i][0] = f[i][1] = 1;
            if (arr[i] > arr[i - 1]) {
                f[i][0] = f[i - 1][1] + 1;
            } else if (arr[i] < arr[i - 1]) {
                f[i][1] = f[i - 1][0] + 1;
            }
            ans = Math.max(ans, Math.max(f[i][0], f[i][1]));
        }
        return ans;
    }
}
