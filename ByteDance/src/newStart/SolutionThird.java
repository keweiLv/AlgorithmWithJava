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
        for (int i = 0;i<arr.length;i++){
            while (arr[i] > stack.peekLast()){
                ans += stack.pollLast() * Math.min(stack.peekLast(),arr[i]);
            }
            stack.offerLast(arr[i]);
        }
        while (stack.size() > 2){
            ans += stack.pollLast() * stack.peekLast();
        }
        return ans;
    }
}
