package newStart;

import java.util.*;

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

    /**
     * 可被三整除的最大和
     * f[0],f[1],f[2]分别代表前 n-1 项取余为 0，为 1，为 2 的情况
     */
    public int maxSumDivThree(int[] nums) {
        final int inf = 1 << 30;
        int[] f = new int[]{0, -inf, -inf};
        for (int num : nums) {
            int[] g = f.clone();
            for (int j = 0; j < 3; j++) {
                g[j] = Math.max(f[j], f[(j - num % 3 + 3) % 3] + num);
            }
            f = g;
        }
        return f[0];
    }

    // 有效的括号字符串
    public boolean checkValidString(String s) {
        Deque<Integer> leftStack = new LinkedList<>();
        Deque<Integer> startStack = new LinkedList<>();
        int n = s.length();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (c == '(') {
                leftStack.push(i);
            } else if (c == '*') {
                startStack.push(i);
            } else {
                if (!leftStack.isEmpty()) {
                    leftStack.pop();
                } else if (!startStack.isEmpty()) {
                    startStack.pop();
                } else {
                    return false;
                }
            }
        }
        while (!leftStack.isEmpty() && !startStack.isEmpty()) {
            int leftIndex = leftStack.pop();
            int startIndex = startStack.pop();
            if (leftIndex > startIndex) {
                return false;
            }
        }
        return leftStack.isEmpty();
    }

    // 骑士在棋盘上的概率
    int[][] dirs = new int[][]{{-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {-2, 1}, {-2, -1}, {2, 1}, {2, -1}};

    public double knightProbability(int n, int k, int row, int column) {
        double[][][] f = new double[n][n][k + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                f[i][j][0] = 1;
            }
        }
        for (int p = 1; p <= k; p++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int[] d : dirs) {
                        int nx = i + d[0], ny = j + d[1];
                        if (nx < 0 || nx >= n || ny < 0 || ny >= n) {
                            continue;
                        }
                        f[i][j][p] += f[nx][ny][p - 1] / 8;
                    }
                }
            }
        }
        return f[row][column][k];
    }

    // 消除游戏
    public int lastRemaining(int n) {
        int head = 1;
        int step = 1;
        boolean left = true;
        while (n > 1) {
            if (left || n % 2 == 1) {
                head += step;
            }
            step *= 2;
            left = !left;
            n /= 2;
        }
        return head;
    }

    // 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        return Math.abs(depth(root.left) - depth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }

    private int depth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(depth(root.left), depth(root.right)) + 1;
    }

    // 找出游戏的获胜者
    public int findTheWinner(int n, int k) {
        int pos = 0;
        for (int i = 2; i < n + 1; i++) {
            pos = (pos + k) % i;
        }
        return pos + 1;
    }

    //水域大小
    public int[] pondSizes(int[][] land) {
        int n = land.length, m = land[0].length;
        var ans = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (land[i][j] == 0) {
                    ans.add(landDfs(land, i, j));
                }
            }
        }
        return ans.stream().sorted().mapToInt(i -> i).toArray();
    }

    private Integer landDfs(int[][] land, int x, int y) {
        if (x < 0 || x >= land.length || y < 0 || y >= land[0].length || land[x][y] != 0) {
            return 0;
        }
        land[x][y] = 1;
        int cnt = 1;
        for (int i = x - 1; i <= x + 1; i++) {
            for (int j = y - 1; j <= y + 1; j++) {
                cnt += landDfs(land, i, j);
            }
        }
        return cnt;
    }

    // 数组中字符串的最大值
    public int maximumValue(String[] strs) {
        int ans = 0;
        for (String str : strs) {
            boolean flag = true;
            for (int i = 0; i < str.length(); i++) {
                if (Character.isDigit(str.charAt(i))) {
                    continue;
                } else {
                    flag = false;
                }
            }
            ans = Math.max(ans, flag ? Integer.parseInt(str) : str.length());
        }
        return ans;
    }

    // 排序数组中只出现一次的数字
    public int singleNonDuplicate(int[] nums) {
        int ans = 0;
        for (int num : nums) {
            ans ^= num;
        }
        return ans;
    }

    // 两两交换链表中的节点
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;
        head.next = swapPairs(next.next);
        next.next = head;
        return next;
    }

    // 优势洗牌
    public int[] advantageCount(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        int n = nums2.length;
        Integer[] idx = new Integer[n];
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> nums2[a] - nums2[b]);
        int left = 0, right = n - 1;
        for (int num : nums1) {
            int i = num > nums2[idx[left]] ? idx[left++] : idx[right--];
            ans[i] = num;
        }
        return ans;
    }

    // 二分查找
    // 循环不变量：nums[left-1] < target,nums[right+1] >= target
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        System.out.println("l:" + left);
        return left < nums.length && nums[left] == target ? left : -1;
    }

}
