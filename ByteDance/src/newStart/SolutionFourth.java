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

    // 两数之和二
    public ListNode addTwoNumbersTwo(ListNode l1, ListNode l2) {
        l1 = reverseList(l1);
        l2 = reverseList(l2);
        ListNode l3 = addTwo(l1, l2, 0);
        return reverseList(l3);
    }

    private ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode nextHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return nextHead;
    }

    private ListNode addTwo(ListNode l1, ListNode l2, int carry) {
        if (l1 == null && l2 == null) {
            return carry != 0 ? new ListNode(carry) : null;
        }
        if (l1 == null) {
            l1 = l2;
            l2 = null;
        }
        carry += l1.val + (l2 != null ? l2.val : 0);
        l1.val = carry % 10;
        l1.next = addTwo(l1.next, (l2 != null ? l2.next : null), carry / 10);
        return l1;
    }

    // 矩阵中的和
    public int matrixSum(int[][] nums) {
        for (int[] num : nums) {
            Arrays.sort(num);
        }
        int ans = 0;
        for (int i = 0; i < nums[0].length; i++) {
            int mx = 0;
            for (int[] num : nums) {
                mx = Math.max(mx, num[i]);
            }
            ans += mx;
        }
        return ans;
    }

    // 航班预定统计
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] ans = new int[n];
        for (int[] book : bookings) {
            int first = book[0], last = book[1], seats = book[2];
            for (int i = first - 1; i < last; i++) {
                ans[i] += seats;
            }
        }
        return ans;
    }

    // k件物品的最大和
    public int kItemsWithMaximumSum(int numOnes, int numZeros, int numNegOnes, int k) {
        return Math.min(numOnes, k) - Math.max(0, k - numOnes - numZeros);
    }

    // 前 k 个高频元素
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> counter = new HashMap<>();
        for (int num : nums) {
            counter.put(num, counter.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> counter.get(a) - counter.get(b));
        counter.forEach((num, cnt) -> {
            if (pq.size() < k) {
                pq.offer(num);
            } else if (counter.get(pq.peek()) < cnt) {
                pq.poll();
                pq.offer(num);
            }
        });
        int[] ans = new int[k];
        int idx = 0;
        for (int num : pq) {
            ans[idx++] = num;
        }
        return ans;
    }

    // 拆分成数目最多的偶整数之和
    public List<Long> maximumEvenSplit(long finalSum) {
        List<Long> ans = new ArrayList<>();
        if (finalSum % 2 == 1) {
            return ans;
        }
        for (long i = 2; i <= finalSum; i += 2) {
            ans.add(i);
            finalSum -= i;
        }
        ans.set(ans.size() - 1, ans.get(ans.size() - 1) + finalSum);
        return ans;
    }

    // 连续子数组的最大和
    public int maxSubArray(int[] nums) {
        int ans = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nums[i] += Math.max(0, nums[i - 1]);
            ans = Math.max(ans, nums[i]);
        }
        return ans;
    }

    // 螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        LinkedList<Integer> res = new LinkedList<>();
        if (matrix == null || matrix.length == 0) {
            return res;
        }
        int left = 0;
        int right = matrix[0].length - 1;
        int top = 0;
        int bottom = matrix.length - 1;
        int eleNum = matrix.length * matrix[0].length;
        while (eleNum >= 1) {
            for (int i = left; i <= right && eleNum >= 1; i++) {
                res.add(matrix[top][i]);
                eleNum--;
            }
            top++;
            for (int i = top; i <= bottom && eleNum >= 1; i++) {
                res.add(matrix[i][right]);
                eleNum--;
            }
            right--;
            for (int i = right; i >= left && eleNum >= 1; i--) {
                res.add(matrix[bottom][i]);
                eleNum--;
            }
            bottom--;
            for (int i = bottom; i >= top && eleNum >= 1; i--) {
                res.add(matrix[i][left]);
                eleNum--;
            }
            left++;
        }
        return res;
    }

    // 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int len = nums.length;
        if (nums == null || nums.length < 3) {
            return res;
        }
        Arrays.sort(nums);
        for (int i = 0; i < len; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int l = i + 1;
            int r = len - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum == 0) {
                    res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while (l < r && nums[l] == nums[l + 1]) {
                        l++;
                    }
                    while (l < r && nums[r] == nums[r - 1]) {
                        r--;
                    }
                    l++;
                    r++;
                } else if (sum < 0) {
                    l++;
                } else if (sum > 0) {
                    r--;
                }
            }
        }
        return res;
    }

    // 二叉树剪枝
    public TreeNode pruneTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);
        if (root.left == null && root.right == null && root.val == 0) {
            return null;
        }
        return root;
    }

    // 山峰数组的顶部
    public int peakIndexInMountainArray(int[] arr) {
        int l = 1, r = arr.length - 2;
        while (l < r) {
            int mid = (l + r) / 2;
            if (arr[mid] > arr[mid - 1] && arr[mid] > arr[mid + 1]) {
                return mid;
            } else if (arr[mid] > arr[mid - 1]) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }

    // 最解决的三数之和
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length; i++) {
            int start = i + 1, end = nums.length - 1;
            while (start < end) {
                int sum = nums[i] + nums[start] + nums[end];
                if (Math.abs(target - sum) < Math.abs(target - ans)) {
                    ans = sum;
                }
                if (sum > target) {
                    end--;
                } else if (sum < target) {
                    start++;
                } else {
                    return ans;
                }
            }
        }
        return ans;
    }

    // 二进制加法
    public String addBinary(String a, String b) {
        int m = a.length() - 1;
        int n = b.length() - 1;
        int carry = 0;
        StringBuilder sb = new StringBuilder();
        while (m >= 0 || n >= 0 || carry != 0) {
            int sum = carry;
            if (m >= 0) {
                sum += a.charAt(m) - '0';
                m--;
            }
            if (n >= 0) {
                sum += b.charAt(n) - '0';
                n--;
            }
            sb.append(sum % 2);
            carry = sum / 2;
        }
        return sb.reverse().toString();
    }

    // 最大子序列交替和
    public long maxAlternatingSum(int[] nums) {
        long f = 0, g = 0;
        for (int x : nums) {
            long ff = Math.max(g - x, f);
            long gg = Math.max(f + x, g);
            f = ff;
            g = gg;
        }
        return Math.max(f, g);
    }

    // 杨辉三角二
    public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i <= rowIndex; i++) {
            List<Integer> cur = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    cur.add(1);
                } else {
                    cur.add(res.get(j - 1) + res.get(j));
                }
            }
            res = cur;
        }
        return res;
    }

    // 交替数字和
    public int alternateDigitSum(int n) {
        int ans = 0, sign = 1;
        for (char c : String.valueOf(n).toCharArray()) {
            int x = c - '0';
            ans += sign * x;
            sign *= -1;
        }
        return ans;
    }

    // 排序数组中两个数字之和
    public int[] twoSum(int[] numbers, int target) {
        int l = 0, r = numbers.length - 1;
        while (l < r) {
            if (numbers[l] + numbers[r] < target) {
                l++;
            } else if (numbers[l] + numbers[r] > target) {
                r--;
            } else {
                return new int[]{l, r};
            }
        }
        return new int[0];
    }

    // 后缀表达式
    public int evalRPN(String[] tokens) {
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < tokens.length; i++) {
            String s = tokens[i];
            if (s.equals("+")) {
                int b = deque.pop();
                int a = deque.pop();
                deque.push(a + b);
                continue;
            }
            if (s.equals("-")) {
                int b = deque.pop();
                int a = deque.pop();
                deque.push(a - b);
                continue;
            }
            if (s.equals("*")) {
                int b = deque.pop();
                int a = deque.pop();
                deque.push(a * b);
                continue;
            }
            if (s.equals("/")) {
                int b = deque.pop();
                int a = deque.pop();
                deque.push(a / b);
                continue;
            }
            deque.push(Integer.valueOf(s));
        }
        return deque.pop();
    }

    // 小行星碰撞
    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> deque = new ArrayDeque<>();
        for (int aster : asteroids) {
            boolean alive = true;
            while (alive && aster < 0 && !deque.isEmpty() && deque.peek() > 0) {
                alive = deque.peek() < -aster;
                if (deque.peek() <= -aster) {
                    deque.pop();
                }
            }
            if (alive) {
                deque.push(aster);
            }
        }
        int size = deque.size();
        int[] ans = new int[size];
        for (int i = size - 1; i >= 0; i--) {
            ans[i] = deque.pop();
        }
        return ans;
    }
}
