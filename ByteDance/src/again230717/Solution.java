package again230717;

import java.util.*;

/**
 * 2023-07-17 another again
 */
public class Solution {


    public static void main(String[] args) {
    }

    // 字符串相加
    public static String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int m = num1.length() - 1, n = num2.length() - 1, carry = 0;
        while (m >= 0 || n >= 0 || carry > 0) {
            int sum = (m >= 0 ? num1.charAt(m) - '0' : 0) + (n >= 0 ? num2.charAt(n) - '0' : 0) + carry;
            carry = sum / 10;
            sum = sum % 10;
            sb.append(sum);
            m--;
            n--;
        }
        return sb.reverse().toString();
    }

    // 分割等和子集
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if ((sum & 1) == 1) {
            return false;
        }
        int target = sum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            for (int j = target; j >= num; j--) {
                dp[j] |= dp[j - num];
            }
        }
        return dp[target];
    }

    // 狒狒吃香蕉
    public int minEatingSpeed(int[] piles, int h) {
        int l = 1, r = Arrays.stream(piles).max().getAsInt();
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (getTime(piles, mid) > h) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }

    private int getTime(int[] piles, int k) {
        int cnt = 0;
        for (int p : piles) {
            cnt += p / k;
            if (p % k > 0) {
                cnt++;
            }
        }
        return cnt;
    }

    // 展平二叉搜索树
    public TreeNode increasingBST(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inOrder(root, res);
        TreeNode dummy = new TreeNode(-1);
        TreeNode cur = dummy;
        for (int val : res) {
            cur.right = new TreeNode(val);
            cur = cur.right;
        }
        return dummy.right;
    }

    private void inOrder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inOrder(root.left, res);
        res.add(root.val);
        inOrder(root.right, res);
    }

    // 数组中和为0 的三个数
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int j = i + 1, k = n - 1;
            while (j < k) {
                while (j > i + 1 && j < n && nums[j] == nums[j - 1]) {
                    j++;
                }
                if (j >= k) {
                    break;
                }
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0) {
                    ans.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    j++;
                } else if (sum > 0) {
                    k--;
                } else {
                    j++;
                }
            }
        }
        return ans;
    }

    // 二叉树的右侧视图
    List<Integer> ans = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        dfs(root, 0);
        return ans;
    }

    private void dfs(TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (depth == ans.size()) {
            ans.add(root.val);
        }
        dfs(root.right, depth + 1);
        dfs(root.left, depth + 1);
    }

    // 宝石与石头
    public int numJewelsInStones(String jewels, String stones) {
        int ans = 0;
        Set<Character> set = new HashSet<>();
        int m = jewels.length(), n = stones.length();
        for (int i = 0; i < m; i++) {
            char c = jewels.charAt(i);
            set.add(c);
        }
        for (int i = 0; i < n; i++) {
            char c = stones.charAt(i);
            if (set.contains(c)) {
                ans++;
            }
        }
        return ans;
    }

    // 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        Deque<Integer> deque = new ArrayDeque<>();
        int n = temperatures.length;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            while (!deque.isEmpty() && temperatures[deque.peek()] < temperatures[i]) {
                int idx = deque.pop();
                ans[idx] = i - idx;
            }
            deque.push(i);
        }
        return ans;
    }

    // 将数组和减半的最少操作次数
    public int halveArray(int[] nums) {
        double s = 0;
        PriorityQueue<Double> pq = new PriorityQueue<>((a, b) -> Double.compare(b, a));
        for (int v : nums) {
            pq.offer(v * 1.0);
            s += v;
        }
        s /= 2.0;
        int ans = 0;
        while (s > 0) {
            double t = pq.poll();
            s -= t / 2.0;
            pq.offer(t / 2.0);
            ++ans;
        }
        return ans;
    }

    // 解决智力问题
    public long mostPoints(int[][] questions) {
        var n = questions.length;
        long[] f = new long[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            int[] q = questions[i];
            int j = i + q[1] + 1;
            f[i] = Math.max(f[i + 1], q[0] + (j < n ? f[j] : 0));
        }
        return f[0];
    }

    // 环形链表
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast.next == null || fast.next.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    // 环形链表二
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
            return null;
        }
        ListNode slow = head.next;
        ListNode fast = head.next.next;
        while (slow != fast) {
            if (fast.next == null || fast.next.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    // 爬楼梯
    public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int[] f = new int[n + 1];
        f[1] = 1;
        f[2] = 2;
        for (int i = 3; i <= n; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[n];
    }

    // 零钱兑换
    int INF = Integer.MAX_VALUE;

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        // 除以二是为了防止下面+1 后溢出
        Arrays.fill(dp, INF / 2);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == INF / 2 ? -1 : dp[amount];
    }

    // 重排链表
    public void reorderList(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode cur = slow.next;
        slow.next = null;
        // 反转 cur
        ListNode pre = null;
        while (cur != null) {
            ListNode t = cur.next;
            cur.next = pre;
            pre = cur;
            cur = t;
        }
        cur = head;
        //合并
        while (pre != null) {
            ListNode t = pre.next;
            pre.next = cur.next;
            cur.next = pre;
            cur = pre.next;
            pre = t;
        }
    }

    // H 指数
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int n = citations.length;
        for (int i = 0; i < n; i++) {
            int remain = n - i;
            if (remain <= citations[i]) {
                return remain;
            }
        }
        return 0;
    }

    // 翻转卡片游戏
    public int flipgame(int[] fronts, int[] backs) {
        Set<Integer> forbidden = new HashSet<>();
        for (int i = 0; i < fronts.length; i++) {
            if (fronts[i] == backs[i]) {
                forbidden.add(fronts[i]);
            }
        }
        int ans = Integer.MAX_VALUE;
        for (int x : fronts) {
            if (!forbidden.contains(x)) {
                ans = Math.min(ans, x);
            }
        }
        for (int x : backs) {
            if (!forbidden.contains(x)) {
                ans = Math.min(ans, x);
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    // 乘积小于 k 的子数组
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int n = nums.length, ans = 0;
        if (k < 1) {
            return 0;
        }
        for (int i = 0, j = 0, cur = 1; i < n; i++) {
            cur *= nums[i];
            while (cur >= k) {
                cur /= nums[j++];
            }
            ans += i - j + 1;
        }
        return ans;
    }

    // 和为 k 的子数组
    public int subarraySum(int[] nums, int k) {
        int n = nums.length, ans = 0;
        int[] sum = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i - 1] + nums[i - 1];
        }
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 1; i <= n; i++) {
            int t = sum[i], d = t - k;
            ans += map.getOrDefault(d, 0);
            map.put(t, map.getOrDefault(t, 0) + 1);
        }
        return ans;
    }

    // 0 和 1个数相同的子数组
    public int findMaxLength(int[] nums) {
        int n = nums.length, ans = 0;
        int[] sum = new int[n + 10];
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i - 1] + (nums[i - 1] == 0 ? -1 : 1);
        }
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 0);
        for (int i = 1; i <= n; i++) {
            int t = sum[i];
            if (map.containsKey(t)) {
                ans = Math.max(ans, i - map.get(t));
            }
            if (!map.containsKey(t)) {
                map.put(t, i);
            }
        }
        return ans;
    }

    // 数字序列中某一位的数字
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        while (n > count) {
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        long num = start + (n - 1) / digit;
        return Long.toString(num).charAt((n - 1) % digit) - '0';
    }

    // 最大单词长度乘积
    public int maxProduct(String[] words) {
        Map<Integer, Integer> map = new HashMap<>();
        for (String w : words) {
            int t = 0, m = w.length();
            for (int i = 0; i < m; i++) {
                int u = w.charAt(i) - 'a';
                t |= (1 << u);
            }
            if (!map.containsKey(t) || map.get(t) < m) {
                map.put(t, m);
            }
        }
        int ans = 0;
        for (int a : map.keySet()) {
            for (int b : map.keySet()) {
                if ((a & b) == 0) {
                    ans = Math.max(ans, map.get(a) * map.get(b));
                }
            }
        }
        return ans;
    }

    // 寻找重复数
    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        slow = nums[slow];
        fast = nums[nums[fast]];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        int pre1 = 0;
        int pre2 = slow;
        while (pre1 != pre2) {
            pre1 = nums[pre1];
            pre2 = nums[pre2];
        }
        return pre1;
    }

    // 找到所有数组中消失的数字
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        for (int num : nums) {
            int x = (num - 1) % n;
            nums[x] += n;
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                res.add(i + 1);
            }
        }
        return res;
    }

    // 缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i] - 1, i);
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    private void swap(int[] nums, int index, int index2) {
        int temp = nums[index];
        nums[index] = nums[index2];
        nums[index2] = temp;
    }

    // 合并两个有序链表
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        } else if (list2 == null) {
            return list1;
        } else if (list1.val < list2.val) {
            list1.next = mergeTwoLists(list1.next, list2);
            return list1;
        } else {
            list2.next = mergeTwoLists(list1, list2.next);
            return list2;
        }
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

    // 合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int len = m + n - 1;
        m--;
        n--;
        while (n >= 0) {
            while (m >= 0 && nums1[m] > nums2[n]) {
                nums1[len--] = nums1[m--];
            }
            nums1[len--] = nums2[n--];
        }
    }

    // 翻转字符串
    public void reverseString(char[] s) {
        int n = s.length;
        for (int i = 0; i < n / 2; i++) {
            char temp = s[i];
            s[i] = s[n - 1 - i];
            s[n - 1 - i] = temp;
        }
    }


    // 搜索推荐系统
    int[][] tr = new int[20010][26];
    int idx = 0;
    Map<Integer, Integer> min = new HashMap<>(), max = new HashMap<>();

    void add(String s, int num) {
        int p = 0;
        for (int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (tr[p][u] == 0) {
                tr[p][u] = ++idx;
                min.put(tr[p][u], num);
            }
            max.put(tr[p][u], num);
            p = tr[p][u];
        }
    }

    int[] query(String s) {
        int a = -1, b = -1, p = 0;
        for (int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (tr[p][u] == 0) {
                return new int[]{-1, -1};
            }
            a = min.get(tr[p][u]);
            b = max.get(tr[p][u]);
            p = tr[p][u];
        }
        return new int[]{a, b};
    }

    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        Arrays.sort(products);
        List<List<String>> ans = new ArrayList<>();
        for (int i = 0; i < products.length; i++) {
            add(products[i], i);
        }
        for (int i = 0; i < searchWord.length(); i++) {
            List<String> list = new ArrayList<>();
            int[] info = query(searchWord.substring(0, i + 1));
            int l = info[0], r = info[1];
            for (int j = l; j <= Math.min(l + 2, r) && l != -1; j++) {
                list.add(products[j]);
            }
            ans.add(list);
        }
        return ans;
    }

    // 数组中两个数的最大异或值
    class Node {
        Node[] ns = new Node[2];
    }

    Node root = new Node();

    void add(int x) {
        Node p = root;
        for (int i = 31; i >= 0; i--) {
            int u = (x >> i) & 1;
            if (p.ns[u] == null) {
                p.ns[u] = new Node();
            }
            p = p.ns[u];
        }
    }

    int getVal(int x) {
        int ans = 0;
        Node p = root;
        for (int i = 31; i >= 0; i++) {
            int a = (x >> i) & 1, b = 1 - a;
            if (p.ns[b] != null) {
                ans |= (b << i);
                p = p.ns[b];
            } else {
                ans |= (a << i);
                p = p.ns[a];
            }
        }
        return ans;
    }

    public int findMaximumXOR(int[] nums) {
        int ans = 0;
        for (int i : nums) {
            add(i);
            int j = getVal(i);
            ans = Math.max(ans, i ^ j);
        }
        return ans;
    }

    // 任意子数组和的绝对值的最大值
    public int maxAbsoluteSum(int[] nums) {
        int ans = 0, fmax = 0, fmin = 0;
        for (int x : nums) {
            fmax = Math.max(fmax, 0) + x;
            fmin = Math.min(fmin, 0) + x;
            ans = Math.max(ans, Math.max(fmax, -fmin));
        }
        return ans;
    }

    // 下一个排列
    public void nextPermutation(int[] nums) {
        int n = nums.length, k = n - 1;
        while (k - 1 >= 0 && nums[k - 1] >= nums[k]) {
            k--;
        }
        if (k == 0) {
            reverse(nums, 0, n - 1);
        } else {
            int u = k;
            while (u + 1 < n && nums[u + 1] > nums[k - 1]) {
                u++;
            }
            swap(nums, k - 1, u);
            reverse(nums, k, n - 1);
        }
    }

    void reverse(int[] nums, int a, int b) {
        int l = a, r = b;
        while (l < r) {
            swap(nums, l++, r--);
        }
    }

//    void swap(int[] nums, int a, int b) {
//        int c = nums[a];
//        nums[a] = nums[b];
//        nums[b] = c;
//    }

    // 颜色分类
    public void sortColors(int[] nums) {
        int n = nums.length;
        int l = 0, r = n - 1, idx = 0;
        while (idx <= r) {
            if (nums[idx] == 0) {
                swap(nums, l++, idx++);
            } else if (nums[idx] == 1) {
                idx++;
            } else {
                swap(nums, idx, r--);
            }
        }
    }

    // 整数的各位积和之差
    public int subtractProductAndSum(int n) {
        int x = 1, y = 0;
        for (; n > 0; n /= 10) {
            int v = n % 10;
            x *= v;
            y += v;
        }
        return x - y;
    }

    // 下降路径最小和二
    public int minFallingPathSum(int[][] grid) {
        int n = grid.length;
        int first_min_sum = 0;
        int second_min_sum = 0;
        int first_min_index = -1;
        for (int i = 0; i < n; i++) {
            int cur_first_min_sum = Integer.MAX_VALUE;
            int cur_second_min_sum = Integer.MAX_VALUE;
            int cur_first_min_index = -1;
            for (int j = 0; j < n; j++) {
                int cur_sum = (j != first_min_index ? first_min_sum : second_min_sum) + grid[i][j];
                if (cur_sum < cur_first_min_sum) {
                    cur_second_min_sum = cur_first_min_sum;
                    cur_first_min_sum = cur_sum;
                    cur_first_min_index = j;
                } else if (cur_sum < cur_second_min_sum) {
                    cur_second_min_sum = cur_sum;
                }
            }
            first_min_sum = cur_first_min_sum;
            second_min_sum = cur_second_min_sum;
            first_min_index = cur_first_min_index;
        }
        return first_min_sum;
    }

    // 戳气球
    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[] temp = new int[n + 2];
        temp[0] = temp[n + 1] = 1;
        for (int i = 0; i < n; i++) {
            temp[i + 1] = nums[i];
        }
        int[][] dp = new int[n + 2][n + 2];
        for (int len = 3; len <= n + 2; len++) {
            for (int l = 0; l + len - 1 <= n + 1; l++) {
                int r = l + len - 1;
                for (int k = l + 1; k <= r - 1; k++) {
                    dp[l][r] = Math.max(dp[l][r], dp[l][k] + dp[k][r] + temp[l] + temp[k] + temp[r]);
                }
            }
        }
        return dp[0][n + 1];
    }

    // 粉刷房子
    public int minCost(int[][] costs) {
        int n = costs.length;
        int a = costs[0][0], b = costs[0][1], c = costs[0][2];
        for (int i = 1; i < n; i++) {
            int d = Math.min(b, c) + costs[i][0];
            int e = Math.min(a, c) + costs[i][1];
            int f = Math.min(a, b) + costs[i][2];
            a = d;
            b = e;
            c = f;
        }
        return Math.min(a, Math.min(b, c));
    }

    // 矩阵对角线元素的和
    public int diagonalSum(int[][] mat) {
        int ans = 0;
        int n = mat.length;
        for (int i = 0; i < n; i++) {
            int j = n - i - 1;
            ans += mat[i][i] + ((i == j) ? 0 : mat[i][j]);
        }
        return ans;
    }

    // 删除链表的倒数第 N 个节点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head;
        ListNode second = dummy;
        for (int i = 0; i < n; i++) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return dummy.next;
    }

    // 循环有序链表的插入
    public again230717.Node insert(again230717.Node head, int insertVal) {
        again230717.Node t = new again230717.Node(insertVal);
        t.next = t;
        if (head == null) {
            return t;
        }
        again230717.Node ans = head;
        int min = head.val, max = head.val;
        while (head.next != ans) {
            head = head.next;
            min = Math.min(min, head.val);
            max = Math.max(max, head.val);
        }
        if (min == max) {
            t.next = ans.next;
            ans.next = t;
        } else {
            while (!(head.val == max && head.next.val == min)) {
                head = head.next;
            }
            while (!(insertVal <= min || insertVal >= max) && !(head.val <= insertVal && insertVal <= head.next.val)) {
                head = head.next;
            }
            t.next = head.next;
            head.next = t;
        }
        return ans;
    }

    // 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return null;
        }
        if (root1 == null || root2 == null) {
            return root1 == null ? root2 : root1;
        }
        root1.val += root2.val;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);
        return root1;
    }

    // 字符串中查找与替换
    public String findReplaceString(String s, int[] indices, String[] sources, String[] targets) {
        int n = s.length();
        String[] replaceStr = new String[n];
        int[] replaceLen = new int[n];
        Arrays.fill(replaceLen, 1);
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (s.startsWith(sources[i], idx)) {
                replaceStr[idx] = targets[i];
                replaceLen[idx] = sources[i].length();
            }
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < n; i += replaceLen[i]) {
            if (replaceStr[i] == null) {
                res.append(s.charAt(i));
            } else {
                res.append(replaceStr[i]);
            }
        }
        return res.toString();
    }

    // 找出转圈游戏输家
    public int[] circularGameLosers(int n, int k) {
        boolean[] vis = new boolean[n];
        int cnt = 0;
        for (int i = 0, p = 1; !vis[i]; p++) {
            vis[i] = true;
            cnt++;
            i = (i + p * k) % n;
        }
        int[] ans = new int[n - cnt];
        for (int i = 0, j = 0; i < n; i++) {
            if (!vis[i]) {
                ans[j++] = i + 1;
            }
        }
        return ans;
    }

    // 二叉树展开为链表
    public void flatten(TreeNode root) {
        List<TreeNode> list = new ArrayList<>();
        preOrderTraversal(root, list);
        int size = list.size();
        for (int i = 1; i < size; i++) {
            TreeNode pre = list.get(i - 1), cur = list.get(i);
            pre.left = null;
            pre.right = cur;
        }
    }

    private void preOrderTraversal(TreeNode root, List<TreeNode> list) {
        if (root != null) {
            list.add(root);
            preOrderTraversal(root.left, list);
            preOrderTraversal(root.right, list);
        }
    }

}
