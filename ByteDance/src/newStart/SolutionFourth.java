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

    // 从尾到头打印链表
    public int[] reversePrint(ListNode head) {
        LinkedList<Integer> stack = new LinkedList<>();
        while (head != null) {
            stack.addLast(head.val);
            head = head.next;
        }
        int n = stack.size();
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            ans[i] = stack.pollLast();
        }
        return ans;
    }

    // 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new LinkedList<>();
        dfs(res, root);
        return res;
    }

    private void dfs(List<Integer> res, TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(res, root.left);
        res.add(root.val);
        dfs(res, root.right);
    }

    // 全排列
    private int[] nums;
    private List<Integer> paths = new ArrayList<>();
    private boolean[] vis;
    private List<List<Integer>> list = new ArrayList<>();

    public List<List<Integer>> permute(int[] nums) {
        this.nums = nums;
        vis = new boolean[nums.length];
        dfs(0);
        return list;
    }

    private void dfs(int i) {
        if (i == nums.length) {
            list.add(new ArrayList<>(paths));
            return;
        }
        for (int j = 0; j < nums.length; j++) {
            if (!vis[j]) {
                paths.add(nums[j]);
                vis[j] = true;
                dfs(i + 1);
                paths.remove(paths.size() - 1);
                vis[j] = false;
            }
        }
    }

    // 找出中枢整数
    public int pivotInteger(int n) {
        for (int i = 1; i <= n; i++) {
            int lSum = (1 + i) * i / 2;
            int rSum = (i + n) * (n - i + 1) / 2;
            if (lSum == rSum) {
                return i;
            }
        }
        return -1;
    }

    // 找出缺失的观测数据
    public int[] missingRolls(int[] rolls, int mean, int n) {
        int m = rolls.length;
        int sum = (n + m) * mean;
        int missing = sum;
        for (int roll : rolls) {
            missing -= roll;
        }
        if (missing < n || missing > n * 6) {
            return new int[0];
        }
        int p = missing / n, remainder = missing % n;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            ans[i] = p + (i < remainder ? 1 : 0);
        }
        return ans;
    }

    // 任务调度器
    public int leastInterval(char[] tasks, int n) {
        int count[] = new int[26];
        for (char task : tasks) {
            count[task - 'A']++;
        }
        Arrays.sort(count);
        int max = count[25];
        int maxNum = 0;
        for (int i = 25; i >= 0; i--) {
            if (count[i] == max) {
                maxNum++;
            } else {
                break;
            }
        }
        return n * (max - 1) <= tasks.length - max - (maxNum - 1) ? tasks.length : (n + 1) * (max - 1) + maxNum;
    }

    // 删除一次得到子数组最大和
    private int[] arr;
    private int[][] memo;

    public int maximumSum(int[] arr) {
        this.arr = arr;
        int ans = Integer.MIN_VALUE, n = arr.length;
        memo = new int[n][2];
        for (int i = 0; i < n; i++) {
            Arrays.fill(memo[i], Integer.MIN_VALUE);
        }
        for (int i = 0; i < n; i++) {
            ans = Math.max(ans, Math.max(maxSumDfs(i, 0), maxSumDfs(i, 1)));
        }
        return ans;
    }

    private int maxSumDfs(int i, int j) {
        if (i < 0) {
            return Integer.MIN_VALUE / 2;
        }
        if (memo[i][j] != Integer.MIN_VALUE) {
            return memo[i][j];
        }
        if (j == 0) {
            return memo[i][j] = Math.max(maxSumDfs(i - 1, 0), 0) + arr[i];
        }
        return memo[i][j] = Math.max(maxSumDfs(i - 1, 1) + arr[i], maxSumDfs(i - 1, 0));
    }

    // 打家劫舍
    public int rob(int[] nums) {
        int pre = 0;
        int cur = 0;
        for (int num : nums) {
            int temp = Math.max(cur, pre + num);
            pre = cur;
            cur = temp;
        }
        return cur;
    }

    // 不同的二叉搜索树二
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new LinkedList<TreeNode>();
        }
        return generateTrees(1, n);
    }

    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> allTrees = new LinkedList<>();
        if (start > end) {
            allTrees.add(null);
            return allTrees;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftTrees = generateTrees(start, i - 1);
            List<TreeNode> rightTrees = generateTrees(i + 1, end);
            for (TreeNode left : leftTrees) {
                for (TreeNode right : rightTrees) {
                    TreeNode cur = new TreeNode(i);
                    cur.left = left;
                    cur.right = right;
                    allTrees.add(cur);
                }
            }
        }
        return allTrees;
    }

    // 最长回文子串
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            --left;
            ++right;
        }
        return right - left - 1;
    }

    // 重构2 行二进制矩阵
    public List<List<Integer>> reconstructMatrix(int upper, int lower, int[] colsum) {
        int n = colsum.length;
        List<Integer> first = new ArrayList<>();
        List<Integer> second = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int a = 0, b = 0;
            if (colsum[i] == 2) {
                a = b = 1;
                upper--;
                lower--;
            } else if (colsum[i] == 1) {
                if (upper > lower) {
                    upper--;
                    a = 1;
                } else {
                    lower--;
                    b = 1;
                }
            }
            if (upper < 0 || lower < 0) {
                break;
            }
            first.add(a);
            second.add(b);
        }
        return upper == 0 && lower == 0 ? List.of(first, second) : List.of();
    }

    // 子集二
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(nums, 0, path, ans);
        return ans;
    }

    private void dfs(int[] nums, int i, List<Integer> path, List<List<Integer>> ans) {
        int n = nums.length;
        if (i == n) {
            ans.add(new ArrayList<>(path));
            return;
        }
        int t = nums[i];
        int last = i;
        while (last < n & nums[last] == nums[i]) {
            last++;
        }
        dfs(nums, last, path, ans);
        for (int j = i; j < last; j++) {
            path.add(nums[j]);
            dfs(nums, last, path, ans);
        }
        for (int j = i; j < last; j++) {
            path.remove(path.size() - 1);
        }
    }

    // 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;
            int sum = x + y + carry;
            carry = sum / 10;
            sum = sum % 10;
            cur.next = new ListNode(sum);
            cur = cur.next;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry == 1) {
            cur.next = new ListNode(carry);
        }
        return pre.next;
    }

    // 查找插入位置
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }
}
