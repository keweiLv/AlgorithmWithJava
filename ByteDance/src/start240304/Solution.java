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

    // 故障键盘
    public String finalString(String s) {
        Deque<Character> q = new ArrayDeque<>();
        boolean tail = true;
        for (char c : s.toCharArray()) {
            if (c == 'i') {
                tail = !tail;
            } else if (tail) {
                q.addLast(c);
            } else {
                q.addFirst(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (char c : q) {
            sb.append(c);
        }
        if (!tail) {
            sb.reverse();
        }
        return sb.toString();
    }

    // 所有可能的二叉树
    private static final List<TreeNode>[] f = new ArrayList[11];

    static {
        Arrays.setAll(f, i -> new ArrayList<>());
        f[1].add(new TreeNode());
        for (int i = 2; i < f.length; i++) { // 计算 f[i]
            for (int j = 1; j < i; j++) { // 枚举左子树叶子数
                for (TreeNode left : f[j]) { // 枚举左子树
                    for (TreeNode right : f[i - j]) { // 枚举右子树
                        f[i].add(new TreeNode(0, left, right));
                    }
                }
            }
        }
    }

    public List<TreeNode> allPossibleFBT(int n) {
        return f[n % 2 > 0 ? (n + 1) / 2 : 0];
    }

    // 找出克隆二叉树中的相同节点
    public final TreeNode getTargetCopy(final TreeNode original, final TreeNode cloned, final TreeNode target) {
        if (original == null || original == target) {
            return cloned;
        }
        TreeNode left = getTargetCopy(original.left, cloned.left, target);
        if (left != null) {
            return left;
        }
        return getTargetCopy(original.right, cloned.right, target);
    }

    // 会议室
    public int minMeetingRooms(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        pq.add(intervals[0][1]);
        for (int i = 1; i < intervals.length; i++) {
            int last = pq.peek();
            if (last <= intervals[i][0]) {
                pq.poll();
                pq.add(intervals[i][1]);
            } else {
                pq.add(intervals[i][1]);
            }
        }
        return pq.size();
    }

    // 有向无环图中一个节点的所有祖先
    public List<List<Integer>> getAncestors(int n, int[][] edges) {
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, i -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[1]].add(e[0]);
        }
        List<Integer>[] ans = new ArrayList[n];
        Arrays.setAll(ans, i -> new ArrayList<>());
        boolean[] vis = new boolean[n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(vis, false);
            dfs(i, g, vis);
            vis[i] = false;
            for (int j = 0; j < n; j++) {
                if (vis[j]) {
                    ans[i].add(j);
                }
            }
        }
        return Arrays.asList(ans);
    }

    private void dfs(int i, List<Integer>[] g, boolean[] vis) {
        vis[i] = true;
        for (int y : g[i]) {
            if (!vis[y]) {
                dfs(y, g, vis);
            }
        }
    }

    // 节点与其祖先之间的最大差值
    private int ans;

    public int maxAncestorDiff(TreeNode root) {
        dfs(root, root.val, root.val);
        return ans;
    }

    private void dfs(TreeNode root, int min, int max) {
        if (root == null) {
            ans = Math.max(ans, max - min);
            return;
        }
        min = Math.min(min, root.val);
        max = Math.max(max, root.val);
        dfs(root.left, min, max);
        dfs(root.right, min, max);
    }

    // 简化路径
    public String simplifyPath(String path) {
        Deque<String> stack = new LinkedList<>();
        String[] split = path.split("/");
        for (String s : split) {
            if (s.equals("..")) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else if (!s.isEmpty() && !s.equals(".")) {
                stack.push(s);
            }
        }
        String res = "";
        for (String d : stack) {
            res = "/" + d + res;
        }
        return res.isEmpty() ? "/" : res;
    }

    // 分发饼干
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int m = g.length, n = s.length;
        int cnt = 0;
        for (int i = 0, j = 0; i < m && j < n; j++, i++) {
            while (j < n && g[i] > s[j]) {
                j++;
            }
            if (j < n) {
                cnt++;
            }
        }
        return cnt;
    }

    // 两整数之和
    public int getSum(int a, int b) {
        while (b != 0) {
            int carry = (a & b) << 1;
            a = a ^ b;
            b = carry;
        }
        return a;
    }

    // 删除有序数组中的重复项
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        int m = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[m++] = nums[i];
            }
        }
        return m;
    }

    // 使数组连续的最少操作数
    public int minOperations(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int m = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[m++] = nums[i];
            }
        }
        int ans = 0;
        int left = 0;
        for (int i = 0; i < m; i++) {
            while (nums[left] < nums[i] - n + 1) {
                left++;
            }
            ans = Math.max(ans, i - left + 1);
        }
        return n - ans;
    }

    // 正整数和负整数的最大计数
    public int maximumCount(int[] nums) {
        int m = 0, n = 0;
        m = (int) Arrays.stream(nums).filter(x -> x < 0).count();
        n = (int) Arrays.stream(nums).filter(x -> x > 0).count();
        return Math.max(m, n);
    }

    // 字符串转换整数
    public int myAtoi(String s) {
        char[] c = s.trim().toCharArray();
        if (c.length == 0) {
            return 0;
        }
        int res = 0, board = Integer.MAX_VALUE / 10;
        int i = 1, sign = 1;
        if (c[0] == '-') {
            sign = -1;
        } else if (c[0] != '+') {
            i = 0;
        }
        for (int j = i; j < c.length; j++) {
            if (c[j] < '0' || c[j] > '9') {
                break;
            }
            if (res > board || res == board && c[j] > '7') {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            res = res * 10 + (c[j] - '0');
        }
        return sign * res;
    }

    // 修改后的最大二进制字符串
    public String maximumBinaryString(String binary) {
        int i = binary.indexOf('0');
        if (i < 0) {
            return binary;
        }
        char[] charArray = binary.toCharArray();
        int cnt = 0;
        for (i++; i < charArray.length; i++) {
            cnt += charArray[i] - '0';
        }
        return "1".repeat(charArray.length - 1 - cnt) + "0" + "1".repeat(cnt);
    }

    // Power(x,n)
    public double myPow(double x, int n) {
        if (x == 0.0f) {
            return 0.0d;
        }
        long b = n;
        double res = 1.0d;
        if (b < 0) {
            x = 1 / x;
            b = -b;
        }
        while (b > 0) {
            if ((b & 1) == 1) {
                res *= x;
            }
            x *= x;
            b >>= 1;
        }
        return res;
    }

    // 全排列二
    boolean[] vis;

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        vis = new boolean[nums.length];
        Arrays.sort(nums);
        backtrace(nums, ans, 0, path);
        return ans;
    }

    private void backtrace(int[] nums, List<List<Integer>> ans, int idx, List<Integer> path) {
        if (idx == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (vis[i] || (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1])) {
                continue;
            }
            path.add(nums[i]);
            vis[i] = true;
            backtrace(nums, ans, idx + 1, path);
            vis[i] = false;
            path.remove(path.size() - 1);
        }
    }

    // 知道冠军一
    public int findChampion(int[][] grid) {
        int ans = 0;
        for (int i = 1; i < grid.length; i++) {
            if (grid[i][ans] == 1) {
                ans = i;
            }
        }
        return ans;
    }

    //  暂时跳过，尽量减少恶意软件的传播
    public int minMalwareSpread(int[][] graph, int[] initial) {
        int n = graph.length;
        boolean[] vis = new boolean[n];
        boolean[] isInit = new boolean[n];
        int mn = Integer.MAX_VALUE;
        for (int x : initial) {
            isInit[x] = true;
            mn = Math.min(mn, x);
        }
        int ans = -1;
        int maxSize = 0;
        for (int x : initial) {
            if (vis[x]) {
                continue;
            }
            nodeId = -1;
            size = 0;
            dfs(x, graph, vis, isInit);
            if (nodeId >= 0 && (size > maxSize || size == maxSize && nodeId < ans)) {
                ans = nodeId;
                maxSize = size;
            }
        }
        return ans < 0 ? mn : ans;
    }

    private int nodeId, size;

    private void dfs(int x, int[][] graph, boolean[] vis, boolean[] isInitial) {
        vis[x] = true;
        size++;
        // 按照状态机更新 nodeId
        if (nodeId != -2 && isInitial[x]) {
            nodeId = nodeId == -1 ? x : -2;
        }
        for (int y = 0; y < graph[x].length; y++) {
            if (graph[x][y] == 1 && !vis[y]) {
                dfs(y, graph, vis, isInitial);
            }
        }
    }

    // 螺旋矩阵
    public int[][] generateMatrix(int n) {
        int l = 0, r = n - 1, t = 0, b = n - 1;
        int[][] mat = new int[n][n];
        int num = 1, tar = n * n;
        while (num <= tar) {
            for (int i = l; i <= r; i++) {
                mat[t][i] = num++;
            }
            t++;
            for (int i = t; i <= b; i++) {
                mat[i][r] = num++;
            }
            r--;
            for (int i = r; i >= l; i--) {
                mat[b][i] = num++;
            }
            b--;
            for (int i = b; i >= t; i--) {
                mat[i][l] = num++;
            }
            l++;
        }
        return mat;
    }

    // 旋转图像
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        // 水平翻转
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - i - 1][j];
                matrix[n - i - 1][j] = temp;
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

    // 从双倍数组中还原原数组
    public int[] findOriginalArray(int[] changed) {
        Arrays.sort(changed);
        int[] ans = new int[changed.length / 2];
        int idx = 0;
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int x : changed) {
            if (!cnt.containsKey(x)) {
                if (idx == ans.length) {
                    return new int[0];
                }
                ans[idx++] = x;
                cnt.merge(x * 2, 1, Integer::sum);
            } else {
                int c = cnt.merge(x, -1, Integer::sum);
                if (c == 0) {
                    cnt.remove(x);
                }
            }
        }
        return ans;
    }

    // 组合总数
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> combine = new ArrayList<>();
        dfs(candidates, target, res, combine, 0);
        return res;
    }

    private void dfs(int[] candidates, int target, List<List<Integer>> res, List<Integer> combine, int idx) {
        if (idx == candidates.length) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList<>(combine));
            return;
        }
        dfs(candidates, target, res, combine, idx + 1);
        if (target - candidates[idx] >= 0) {
            combine.add(candidates[idx]);
            dfs(candidates, target - candidates[idx], res, combine, idx);
            combine.remove(combine.size() - 1);
        }
    }

    // 组合总数四
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int num : nums) {
                if (num <= i) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }

    // 组合总数二
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        Arrays.sort(candidates);
        dfsNew(candidates, target, res, path, 0);
        return res;
    }

    private void dfsNew(int[] candidates, int target, List<List<Integer>> res, List<Integer> path, int idx) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
        }
        Set<Integer> set = new HashSet<>();
        for (int i = idx; i < candidates.length; i++) {
            if (target - candidates[i] < 0) {
                continue;
            }
            if (!set.add(candidates[i])) {
                continue;
            }
            path.add(candidates[i]);
            dfsNew(candidates, target - candidates[i], res, path, idx + 1);
            path.remove(path.size() - 1);
        }
    }

    // 爱生气的书店老板
    public int maxSatisfied(int[] customers, int[] grumpy, int minutes) {
        int n = customers.length, ans = 0;
        for (int i = 0; i < n; i++) {
            if (grumpy[i] == 0) {
                ans += customers[i];
                customers[i] = 0;
            }
        }
        int cur = 0, max = 0;
        for (int i = 0; i < n; i++) {
            cur += customers[i];
            if (i >= minutes) {
                cur -= customers[i - minutes];
            }
            max = Math.max(max, cur);
        }
        return ans + max;
    }

    // 总行驶距离
    public int distanceTraveled(int mainTank, int additionalTank) {
        int ans = 0;
        while (mainTank >= 5) {
            int t = mainTank / 5;
            ans += t * 50;
            mainTank %= 5;
            t = Math.min(t, additionalTank);
            additionalTank -= t;
            mainTank += t;
        }
        return ans + mainTank * 10;
    }

    // 寻找旋转排序数组中的最小值
    public int findMin(int[] nums) {
        int i = 0, j = nums.length - 1;
        while (i < j) {
            int mid = i + (j - i) / 2;
            if (nums[mid] > nums[j]) {
                i = mid + 1;
            } else if (nums[mid] < nums[j]) {
                j = mid;
            } else {
                j--;
            }
        }
        return nums[i];
    }

    // 将矩阵按对角线排序
    public int[][] diagonalSort(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int[] a = new int[Math.min(m, n)];
        for (int k = 1 - n; k < m; k++) {
            int left = Math.max(k, 0);
            int right = Math.min(k + n, m);
            for (int i = left; i < right; i++) {
                a[i - left] = mat[i][i - k];
            }
            Arrays.sort(a, 0, right - left);
            for (int i = left; i < right; i++) {
                mat[i][i - k] = a[i - left];
            }
        }
        return mat;
    }

    // 拆炸弹
    public int[] decrypt(int[] code, int k) {
        int n = code.length;
        int[] ans = new int[n];
        int r = k > 0 ? k + 1 : n;
        k = Math.abs(k);
        int s = 0;
        for (int i = r - k; i < r; i++) {
            s += code[i];
        }
        for (int i = 0; i < n; i++) {
            ans[i] = s;
            s += code[r % n] - code[(r - k) % n];
            r++;
        }
        return ans;
    }

    // 单词搜索
    private static final int[][] dir = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    private int rows;
    private int cols;
    private int len;
    private boolean[][] visited;
    private char[] charArray;
    private char[][] board;

    public boolean exist(char[][] board, String word) {
        rows = board.length;
        if (rows == 0) {
            return false;
        }
        cols = board[0].length;
        visited = new boolean[rows][cols];
        this.len = word.length();
        this.charArray = word.toCharArray();
        this.board = board;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (dfs(i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(int x, int y, int begin) {
        if (begin == len - 1) {
            return board[x][y] == charArray[begin];
        }
        if (board[x][y] == charArray[begin]) {
            visited[x][y] = true;
            for (int[] dir : dir) {
                int newX = x + dir[0];
                int newY = y + dir[1];
                if (inArea(newX, newY) && !visited[newX][newY]) {
                    if (dfs(newX, newY, begin + 1)) {
                        return true;
                    }
                }
            }
            visited[x][y] = false;
        }
        return false;
    }

    private boolean inArea(int newX, int newY) {
        return newX >= 0 && newX < rows && newY >= 0 && newY < cols;
    }

    // 从前序与中序遍历序列构造二叉树
    private Map<Integer, Integer> indexMap;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        indexMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
    }

    private TreeNode myBuildTree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        if (preorder_left > preorder_right) {
            return null;
        }
        int preorder_root = preorder_left;
        int inorder_root = indexMap.get(preorder[preorder_root]);

        TreeNode root = new TreeNode(preorder[preorder_root]);
        int left_size = inorder_root - inorder_left;
        root.left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + left_size, inorder_left, inorder_root - 1);
        root.right = myBuildTree(preorder, inorder, preorder_left + left_size + 1, preorder_right, inorder_root + 1, inorder_right);
        return root;
    }

    // 给植物浇水
    public int wateringPlants(int[] plants, int capacity) {
        int n = plants.length;
        int ans = n;
        int water = capacity;
        for (int i = 0; i < n; i++) {
            if (water < plants[i]) {
                ans += 2 * i;
                water = capacity;
            }
            water -= plants[i];
        }
        return ans;
    }

    // 给植物浇水二
    public int minimumRefill(int[] plants, int capacityA, int capacityB) {
        int ans = 0;
        int a = capacityA;
        int b = capacityB;
        int i = 0;
        int j = plants.length - 1;
        while (i < j) {
            if (a < plants[i]) {
                ans++;
                a = capacityA;
            }
            a -= plants[i++];
            if (b < plants[j]) {
                ans++;
                b = capacityB;
            }
            b -= plants[j--];
        }
        if (i == j && Math.max(a, b) < plants[i]) {
            ans++;
        }
        return ans;
    }

    // 测试已统计设备
    public int countTestedDevices(int[] batteryPercentages) {
        int ans = 0;
        int cnt = 0;
        for (int num : batteryPercentages) {
            num = Math.max(0, num - cnt);
            if (num > 0) {
                ans++;
                cnt++;
            }
        }
        return ans;
    }

    // 腐烂的橘子】
    private static final int[][] DIRECTIONS = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    public int orangesRotting(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int fresh = 0;
        List<int[]> q = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    fresh++;
                } else if (grid[i][j] == 2) {
                    q.add(new int[]{i, j});
                }
            }
        }

        int ans = -1;
        while (!q.isEmpty()) {
            ans++;
            List<int[]> tmp = q;
            q = new ArrayList<>();
            for (int[] pos : tmp) {
                for (int[] d : DIRECTIONS) {
                    int i = pos[0] + d[0];
                    int j = pos[1] + d[1];
                    if (0 <= i && i < m && 0 <= j && j < n && grid[i][j] == 1) {
                        fresh--;
                        grid[i][j] = 2;
                        q.add(new int[]{i, j});
                    }
                }
            }
        }
        return fresh > 0 ? -1 : Math.max(0, ans);
    }

    // 完成所有任务需要的最少轮数
    public int minimumRounds(int[] tasks) {
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int t : tasks) {
            cnt.merge(t, 1, Integer::sum);
        }
        int ans = 0;
        for (int va : cnt.values()) {
            if (va == 1) {
                return -1;
            }
            ans += (va + 2) / 3;
        }
        return ans;
    }

    // 完成所有任务的最少时间
    public int findMinimumTime(int[][] tasks) {
        Arrays.sort(tasks, (a, b) -> a[1] - b[1]);
        int ans = 0;
        int max = tasks[tasks.length - 1][1];
        boolean[] run = new boolean[max + 1];
        for (int[] t : tasks) {
            int start = t[0];
            int end = t[1];
            int d = t[2];
            for (int i = start; i <= end; i++) {
                if (run[i]) {
                    d--;
                }
            }
            for (int i = end; d > 0; i--) {
                if (!run[i]) {
                    run[i] = true;
                    d--;
                    ans++;
                }
            }
        }
        return ans;
    }

    // 最大子数组之和
    public int maxSubArray(int[] nums) {
        int pre = 0, ans = nums[0];
        for (int num : nums) {
            pre = Math.max(pre + num, num);
            ans = Math.max(ans, pre);
        }
        return ans;
    }

    // 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;
        int[] L = new int[len];
        int[] R = new int[len];
        int[] ans = new int[len];
        L[0] = 1;
        for (int i = 1; i < len; i++) {
            L[i] = L[i - 1] * nums[i - 1];
        }
        R[len - 1] = 1;
        for (int i = len - 2; i >= 0; i--) {
            R[i] = R[i + 1] * nums[i + 1];
        }
        for (int i = 0; i < len; i++) {
            ans[i] = L[i] * R[i];
        }
        return ans;
    }

    // 你可以工作的最大周数
    public long numberOfWeeks(int[] milestones) {
        long s = 0;
        int m = 0;
        for (int x : milestones) {
            s += x;
            m = Math.max(x, m);
        }
        return m > s - m + 1 ? (s - m) * 2 + 1 : s;
    }

    // 找出输掉零场或一场比赛的玩家
    public List<List<Integer>> findWinners(int[][] matches) {
        Map<Integer, Integer> losscnt = new HashMap<>();
        for (int[] m : matches) {
            if (!losscnt.containsKey(m[0])) {
                losscnt.put(m[0], 0);
            }
            losscnt.merge(m[1], 1, Integer::sum);
        }
        List<List<Integer>> ans = List.of(new ArrayList<>(), new ArrayList<>());
        for (Map.Entry<Integer, Integer> e : losscnt.entrySet()) {
            int cnt = e.getValue();
            if (cnt < 2) {
                ans.get(cnt).add(e.getKey());
            }
        }
        Collections.sort(ans.get(0));
        Collections.sort(ans.get(1));
        return ans;
    }

    // 分糖果二
    public int[] distributeCandies(int candies, int num_people) {
        int n = num_people;
        int[] ans = new int[n];
        for (int i = 1; candies > 0; i++) {
            ans[(i - 1) % n] += Math.min(i, candies);
            candies -= i;
        }
        return ans;
    }

    // 取整购买后的账户余额
    public int accountBalanceAfterPurchase(int purchaseAmount) {
        return 100 - (purchaseAmount + 5) / 10 * 10;
    }

    // 子序列最大优雅度
    public long findMaximumElegance(int[][] items, int k) {
        Arrays.sort(items, (a, b) -> b[0] - a[0]);
        long ans = 0;
        long totalProfit = 0;
        Set<Integer> vis = new HashSet<>();
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < items.length; i++) {
            int profit = items[i][0];
            int category = items[i][1];
            if (i < k) {
                totalProfit += profit;
                if (!vis.add(category)) {
                    deque.push(profit);
                }
            } else if (!deque.isEmpty() && vis.add(category)) {
                totalProfit += profit - deque.pop();
            }
            ans = Math.max(ans, totalProfit + (long) vis.size() * vis.size());
        }
        return ans;
    }

    // 价格减免
    public String discountPrices(String sentence, int discount) {
        double d = 1 - discount / 100.0;
        String[] a = sentence.split(" ");
        for (int i = 0; i < a.length; i++) {
            if (check(a[i])) {
                a[i] = String.format("$%.2f", Long.parseLong(a[i].substring(1)) * d);
            }
        }
        return String.join(" ", a);
    }

    private boolean check(String s) {
        if (s.length() == 1 || s.charAt(0) != '$') {
            return false;
        }
        char[] cs = s.toCharArray();
        for (int i = 1; i < cs.length; i++) {
            if (!Character.isDigit(cs[i])) {
                return false;
            }
        }
        return true;
    }

    // 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        int res = 0;
        Arrays.fill(dp, 1);
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    // 美丽下标对的数目
    public int countBeautifulPairs(int[] nums) {
        int ans = 0;
        int[] cnt = new int[10];
        for (int x : nums) {
            for (int y = 1; y < 10; y++) {
                if (cnt[y] > 0 && gcd(y,x % 10) == 1){
                    ans += cnt[y];
                }
            }
            while (x >= 10){
                x /= 10;
            }
            cnt[x]++;
        }
        return ans;
    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }
}