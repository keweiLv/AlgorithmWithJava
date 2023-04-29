package LC_RACE.Biweekly_103;

import java.util.Arrays;

/**
 * 第 103 场双周赛
 */
public class Solution {

    // K个元素的最大和
    public int maximizeSum(int[] nums, int k) {
        int n = nums.length;
        int target = Arrays.stream(nums).max().getAsInt();
        int ans = 0;
        for (int i = 0; i < k; i++) {
            ans += target++;
        }
        return ans;
    }

    // 找到两个数组的前缀公共数组
    public int[] findThePrefixCommonArray(int[] a, int[] b) {
        int n = a.length;
        int[] ans = new int[n];
        int[] rec = new int[51];
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            if (rec[a[i]] == 1) {
                cnt++;
            } else {
                rec[a[i]] = 1;
            }
            if (rec[b[i]] == 1) {
                cnt++;
            } else {
                rec[b[i]] = 1;
            }
            ans[i] = cnt;
        }
        return ans;
    }

    // 网格图中鱼的最大树目
    static int maxFish = 0;
    public static int findMaxFish(int[][] grid) {
        int ans = 0;
        int rows = grid.length, columns = grid[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                int tmp = 0;
                dfs(i, j, grid);
                ans = Math.max(ans, maxFish);
                maxFish = 0;
            }
        }
        return ans;
    }

    private static void dfs(int r, int c, int[][] grid) {
        if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length) {
            return;
        }
        if (grid[r][c] == 0) {
            return;
        }
        maxFish += grid[r][c];
        grid[r][c] = 0;
        int[][] drict = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        for (int i = 0; i < drict.length; i++) {
            dfs(r + drict[i][0], c + drict[i][1], grid);
        }
    }

    public static void main(String[] args) {
        int[][] init = new int[][]{{0, 2, 1, 0}, {4, 0, 0, 3}, {1, 0, 0, 4}, {0, 3, 2, 0}};
        System.out.println(findMaxFish(init));
    }
}
