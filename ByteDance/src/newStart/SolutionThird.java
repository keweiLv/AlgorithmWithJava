package newStart;

import java.util.HashMap;
import java.util.Map;

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
}
