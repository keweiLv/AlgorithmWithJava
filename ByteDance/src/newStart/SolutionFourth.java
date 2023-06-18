package newStart;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class SolutionFourth {

    // 二叉树中和为某一值的路径
    LinkedList<List<Integer>> res = new LinkedList<>();
    LinkedList<Integer> path = new LinkedList<>();

    public List<List<Integer>> pathSum(TreeNode root, int target) {
        recur(root, target);
        return res;
    }

    private void recur(TreeNode node, int target) {
        if (node == null) {
            return;
        }
        path.add(node.val);
        target -= node.val;
        if (target == 0 && node.left == null && node.right == null) {
            res.add(new LinkedList<>(path));
        }
        recur(node.left, target);
        recur(node.right, target);
        path.removeLast();
    }

    // 分割圆的最少切割次数
    public int numberOfCuts(int n) {
        return n > 1 && n % 2 == 1 ? n : n >> 1;
    }

    // 二叉树最大宽度
    Map<Integer, Integer> map = new HashMap<>();
    int ans;

    public int widthOfBinaryTree(TreeNode root) {
        dfs(root, 1, 0);
        return ans;
    }

    private void dfs(TreeNode root, int val, int depth) {
        if (root == null) {
            return;
        }
        if (!map.containsKey(depth)) {
            map.put(depth, val);
        }
        ans = Math.max(ans, val - map.get(depth) + 1);
        val = val - map.get(depth) + 1;
        dfs(root.left, val << 1, depth + 1);
        dfs(root.right, val << 1 | 1, depth + 1);
    }

    // 统计封闭岛屿的数目
    private int n, m;
    private int[][] grid;

    public int closedIsland(int[][] grid) {
        n = grid.length;
        m = grid[0].length;
        this.grid = grid;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 0) {
                    ans += dfs(i, j);
                }
            }
        }
        return ans;
    }

    private int dfs(int i, int j) {
        int res = i > 0 && i < n - 1 && j > 0 && j < m - 1 ? 1 : 0;
        grid[i][j] = 1;
        int[][] dirt = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] ints : dirt) {
            int x = i + ints[0], y = j + ints[1];
            if (x >= 0 && x < n && y >= 0 && y < m && grid[x][y] == 0) {
                res &= dfs(x, y);
            }
        }
        return res;
    }
}
