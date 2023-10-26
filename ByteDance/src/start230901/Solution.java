package start230901;

import java.util.*;

public class Solution {

    // 买铅笔和钢笔的方案数
    public long waysToBuyPensPencils(int total, int cost1, int cost2) {
        if (cost1 < cost2) {
            return waysToBuyPensPencils(total, cost2, cost1);
        }
        long res = 0, choose = 0;
        while (choose * cost1 <= total) {
            res += (total - choose * cost1) / cost2 + 1;
            choose++;
        }
        return res;
    }

    // 消灭怪物的最大数量
    public int eliminateMaximum(int[] dist, int[] speed) {
        int n = dist.length;
        int[] arrivalTimes = new int[n];
        for (int i = 0; i < n; i++) {
            arrivalTimes[i] = (dist[i] - 1) / speed[i] + 1;
        }
        Arrays.sort(arrivalTimes);
        for (int i = 0; i < n; i++) {
            if (arrivalTimes[i] <= i) {
                return i;
            }
        }
        return n;
    }

    // 从两个数字数组里生成最小数字
    public int minNumber(int[] nums1, int[] nums2) {
        int ans = 100;
        for (int a : nums1) {
            for (int b : nums2) {
                if (a == b) {
                    ans = Math.min(ans, a);
                } else {
                    ans = Math.min(ans, Math.min(a * 10 + b, b * 10 + a));
                }
            }
        }
        return ans;
    }

    // 最多可以摧毁的敌人堡垒数目
    public int captureForts(int[] forts) {
        int n = forts.length;
        int ans = 0, pre = -1;
        for (int i = 0; i < n; i++) {
            if (forts[i] == 1 || forts[i] == -1) {
                if (pre >= 0 && forts[i] != forts[pre]) {
                    ans = Math.max(ans, i - pre - 1);
                }
                pre = i;
            }
        }
        return ans;
    }

    // 最深叶节点的最近公共祖先
    private TreeNode treeNodeAns;
    private int maxDepth = -1;

    public TreeNode lcaDeepestLeaves(TreeNode root) {
        dfs(root, 0);
        return treeNodeAns;
    }

    private int dfs(TreeNode root, int depth) {
        if (root == null) {
            maxDepth = Math.max(maxDepth, depth);
            return depth;
        }
        int leftDepth = dfs(root.left, depth + 1);
        int rightDepth = dfs(root.right, depth + 1);
        if (leftDepth == rightDepth && leftDepth == maxDepth) {
            treeNodeAns = root;
        }
        return Math.max(leftDepth, rightDepth);
    }

    // 修车的最少时间
    public long repairCars(int[] ranks, int cars) {
        long l = 0, r = 1L * ranks[0] * cars * cars;
        while (l < r) {
            long m = l + r >> 1;
            if (check(ranks, cars, m)) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }

    private boolean check(int[] ranks, int cars, long m) {
        long cnt = 0;
        for (int x : ranks) {
            cnt += (long) Math.sqrt(m / x);
        }
        return cnt >= cars;
    }

    // 每个小孩最多能分到多少糖果
    public int maximumCandies(int[] candies, long k) {
        int max = Arrays.stream(candies).max().getAsInt();
        int left = 0, right = max;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (getKids(candies, mid) < k) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return left;
    }

    private long getKids(int[] candies, int mid) {
        long cnt = 0;
        for (int candy : candies) {
            cnt += candy / mid;
        }
        return cnt;
    }

    // 计算列车到站时间
    public int findDelayedArrivalTime(int arrivalTime, int delayedTime) {
        return (arrivalTime + delayedTime) % 24;
    }

    // 最长回文串
    public int longestPalindrome(String s) {
        int[] arr = new int[128];
        for (char c : s.toCharArray()) {
            arr[c]++;
        }
        int count = 0;
        for (int i : arr) {
            count += (i % 2);
        }
        return count == 0 ? s.length() : (s.length() - count + 1);
    }

    // 多个数组求交集
    public List<Integer> intersection(int[][] nums) {
        int[] cnt = new int[10001];
        int n = nums.length;
        for (int[] num : nums) {
            for (int x : num) {
                cnt[x]++;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i <= 1000; i++) {
            if (cnt[i] == n) {
                ans.add(i);
            }
        }
        return ans;
    }

    // 课程表
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses <= 0) {
            return new int[0];
        }
        HashSet<Integer>[] adj = new HashSet[numCourses];
        for (int i = 0; i < numCourses; i++) {
            adj[i] = new HashSet<>();
        }
        int[] inDegree = new int[numCourses];
        for (int[] p : prerequisites) {
            adj[p[1]].add(p[0]);
            inDegree[p[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        int[] res = new int[numCourses];
        int count = 0;
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();
            res[count++] = poll;
            Set<Integer> tmp = adj[poll];
            for (Integer num : tmp) {
                inDegree[num]--;
                if (inDegree[num] == 0) {
                    queue.offer(num);
                }
            }
        }
        if (count == numCourses) {
            return res;
        }
        return new int[0];
    }

    // 最大子数组和
    public int maxSubArray(int[] nums) {
        int ans = nums[0], pre = nums[0];
        for (int i = 1; i < nums.length; i++) {
            pre = Math.max(pre + nums[i], nums[i]);
            ans = Math.max(ans, pre);
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

    // 课程表四
    public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
        boolean[][] f = new boolean[numCourses][numCourses];
        List<Integer>[] g = new List[numCourses];
        int[] indeg = new int[numCourses];
        Arrays.setAll(g, i -> new ArrayList<>());
        for (int[] p : prerequisites) {
            g[p[0]].add(p[1]);
            ++indeg[p[1]];
        }
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) {
            if (indeg[i] == 0) {
                deque.offer(i);
            }
        }
        while (!deque.isEmpty()) {
            Integer i = deque.poll();
            for (int j : g[i]) {
                f[i][j] = true;
                for (int h = 0; h < numCourses; h++) {
                    f[h][j] |= f[h][i];
                }
                if (--indeg[j] == 0) {
                    deque.offer(j);
                }
            }
        }
        List<Boolean> ans = new ArrayList<>();
        for (int[] q : queries) {
            ans.add(f[q[0]][q[1]]);
        }
        return ans;
    }

    // 子集二
    List<List<Integer>> ans;
    List<Integer> path;

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        ans = new ArrayList<>();
        path = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        boolean[] vis = new boolean[n];
        backtrace(nums, 0, vis, 0);
        return ans;
    }

    private void backtrace(int[] nums, int start, boolean[] vis, int n) {
        ans.add(new ArrayList<>(path));
        for (int i = start; i < n; i++) {
            if (i > 0 && nums[i - 1] == nums[i] && !vis[i - 1]) {
                continue;
            }
            vis[i] = true;
            path.add(nums[i]);
            backtrace(nums, i + 1, vis, n);
            vis[i] = false;
            path.remove(path.size() - 1);
        }
    }

    // 可以攻击国王的皇后
    private final static int[][] directions = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};

    public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
        boolean[][] check = new boolean[8][8];
        for (int[] q : queens) {
            check[q[0]][q[1]] = true;
        }
        List<List<Integer>> ans = new ArrayList<>();
        for (int[] d : directions) {
            int x = king[0] + d[0];
            int y = king[1] + d[1];
            while (0 <= x && x < 8 && 0 <= y && y < 8) {
                if (check[x][y]) {
                    ans.add(List.of(x, y));
                    break;
                }
                x += d[0];
                y += d[1];
            }
        }
        return ans;
    }

    // 划分字母区间
    public List<Integer> partitionLabels(String s) {
        List<Integer> ans = new ArrayList<>();
        int n = s.length();
        int[] maxIndex = new int[26];
        for (int i = 0; i < n; i++) {
            maxIndex[s.charAt(i) - 'a'] = i;
        }
        int start = 0, end = 0;
        for (int i = 0; i < n; i++) {
            end = Math.max(end, maxIndex[s.charAt(i) - 'a']);
            if (i == end) {
                ans.add(end - start + 1);
                start = end + 1;
            }
        }
        return ans;
    }

    // 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode cur = head, pre = null;
        while (cur != null) {
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        return pre;
    }

    // 宝石补给
    public int giveGem(int[] gem, int[][] operations) {
        for (int[] op : operations) {
            int tmp = gem[op[0]] / 2;
            gem[op[0]] -= tmp;
            gem[op[1]] += tmp;
        }
        return Arrays.stream(gem).max().getAsInt() - Arrays.stream(gem).min().getAsInt();
    }

    // 单词的压缩编码
    Trie trie = new Trie();

    public int minimumLengthEncoding(String[] words) {
        for (String word : words) {
            trie.insert(word);
        }
        int res = 0;
        Set<String> set = new HashSet<>();
        for (String word : words) {
            if (set.contains(word)) {
                continue;
            }
            set.add(word);
            boolean maxWord = trie.search(word);
            if (maxWord) {
                res = res + word.length() + 1;
            }
        }
        return res;
    }

    class Trie {
        TrieNode root = new TrieNode();

        public void insert(String word) {
            TrieNode cur = root;
            boolean newBranch = false;
            for (int i = word.length() - 1; i >= 0; i--) {
                char c = word.charAt(i);
                if (cur.next[c - 'a'] == null) {
                    cur.next[c - 'a'] = new TrieNode();
                    newBranch = true;
                }
                cur = cur.next[c - 'a'];
                if (cur.isEnd && i != 0) {
                    cur.isEnd = false;
                }
            }
            if (newBranch) {
                cur.isEnd = true;
            }
        }

        public boolean search(String word) {
            TrieNode cur = root;
            for (int i = word.length() - 1; i >= 0; i--) {
                char c = word.charAt(i);
                if (cur.next[c - 'a'] == null) {
                    return false;
                }
                cur = cur.next[c - 'a'];
            }
            return cur.isEnd;
        }
    }

    class TrieNode {
        TrieNode[] next = new TrieNode[26];
        boolean isEnd;
    }

    // 打家劫舍
    public int rob(int[] nums) {
        int pre = 0, cur = 0;
        for (int num : nums) {
            int tmp = Math.max(cur, pre + num);
            pre = cur;
            cur = tmp;
        }
        return cur;
    }

    // 打家劫舍二
    public int robG2(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)), myRob(Arrays.copyOfRange(nums, 1, nums.length)));
    }

    public int myRob(int[] nums) {
        int pre = 0, cur = 0;
        for (int num : nums) {
            int tmp = Math.max(pre + num, cur);
            pre = cur;
            cur = tmp;
        }
        return cur;
    }

    // 七进制数
    public String convertToBase7(int num) {
        boolean flag = num < 0;
        if (flag) {
            num = -num;
        }
        StringBuffer sb = new StringBuffer();
        do {
            sb.append(num % 7);
            num /= 7;
        } while (num != 0);
        sb.reverse();
        return flag ? "-" + sb.toString() : sb.toString();
    }

    // 拿硬币
    public int minCount(int[] coins) {
        int ans = 0;
        for (int num : coins) {
            ans += (num + 1) / 2;
        }
        return ans;
    }

    // 二叉数的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return Math.max(left, right) + 1;
    }

    // 将钱分给最多的儿童
    public int distMoney(int money, int children) {
        if (money < children) {
            return -1;
        }
        if (money > 8 * children) {
            return children - 1;
        }
        if (money == 8 * children - 4) {
            return children - 2;
        }
        return (money - children) / 7;
    }

    // 递枕头
    public int passThePillow(int n, int time) {
        int k = time / (n - 1);
        int mod = time % (n - 1);
        return (k & 1) == 1 ? n - mod : mod + 1;
    }

    // 餐厅过滤器
    public List<Integer> filterRestaurants(int[][] restaurants, int veganFriendly, int maxPrice, int maxDistance) {
        Arrays.sort(restaurants, (a, b) -> a[1] == b[1] ? b[0] - a[0] : b[1] - a[1]);
        List<Integer> ans = new ArrayList<>();
        for (int[] r : restaurants) {
            if (r[2] >= veganFriendly && r[3] <= maxPrice && r[4] <= maxDistance) {
                ans.add(r[0]);
            }
        }
        return ans;
    }

    // 最小和分割
    public int splitNum(int num) {
        char[] str = Integer.toString(num).toCharArray();
        Arrays.sort(str);
        int num1 = 0, num2 = 0;
        for (int i = 0; i < str.length; i++) {
            if (i % 2 == 0) {
                num1 = num1 * 10 + (str[i] - '0');
            } else {
                num2 = num2 * 10 + (str[i] - '0');
            }
        }
        return num1 + num2;
    }

    // 移动机器人
    public int sumDistance(int[] nums, String s, int d) {
        final long MOD = (long) 1e9 + 7;
        int n = nums.length;
        long[] a = new long[n];
        for (int i = 0; i < n; i++) {
            a[i] = (long) nums[i] + (s.charAt(i) == 'L' ? -d : d);
        }
        Arrays.sort(a);
        long ans = 0, sum = 0;
        for (int i = 0; i < n; i++) {
            ans = (ans + i * a[i] - sum) % MOD;
            sum += a[i];
        }
        return (int) ans;
    }

    // 奖励最顶尖的K名学生
    public List<Integer> topStudents(String[] positive_feedback, String[] negative_feedback, String[] report, int[] student_id, int k) {
        Set<String> ps = new HashSet<>(Arrays.asList(positive_feedback));
        Set<String> ns = new HashSet<>(Arrays.asList(negative_feedback));
        int n = report.length;
        int[][] arr = new int[n][2];
        for (int i = 0; i < n; i++) {
            int sid = student_id[i];
            int s = 0;
            for (String t : report[i].split(" ")) {
                if (ps.contains(t)) {
                    s += 3;
                } else if (ns.contains(t)) {
                    s -= 1;
                }
            }
            arr[i] = new int[]{s, sid};
        }
        Arrays.sort(arr, (a, b) -> a[0] == b[0] ? a[1] - b[1] : b[0] - a[0]);
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            ans.add(arr[i][1]);
        }
        return ans;
    }

    // 找出数组的串联值
    public long findTheArrayConcVal(int[] nums) {
        long ans = 0;
        int i = 0, j = nums.length - 1;
        for (; i < j; i++, j--) {
            ans += Integer.parseInt(nums[i] + "" + nums[j]);
        }
        if (i == j) {
            ans += nums[i];
        }
        return ans;
    }

    // 避免洪水泛滥
    public int[] avoidFlood(int[] rains) {
        int n = rains.length;
        int[] ans = new int[n];
        Arrays.fill(ans, -1);
        TreeSet<Integer> sunny = new TreeSet<>();
        Map<Integer, Integer> rainy = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int rain = rains[i];
            if (rain > 0) {
                if (rainy.containsKey(rain)) {
                    Integer higher = sunny.higher(rainy.get(rain));
                    if (higher == null) {
                        return new int[0];
                    }
                    ans[higher] = rain;
                    sunny.remove(higher);
                }
                rainy.put(rain, i);
            } else {
                sunny.add(i);
                ans[i] = 1;
            }
        }
        return ans;
    }


    // 只出现一次的数字二
    public int singleNumber(int[] nums) {
        int[] counts = new int[32];
        for (int num : nums) {
            for (int j = 0; j < 32; j++) {
                counts[j] += num & 1;
                num >>>= 1;
            }
        }
        int res = 0, m = 3;
        for (int i = 0; i < 32; i++) {
            res <<= 1;
            res |= counts[31 - i] % m;
        }
        return res;
    }

    // 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        int[] table = new int[26];
        for (int i = 0; i < s.length(); i++) {
            table[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i < t.length(); i++) {
            table[t.charAt(i) - 'a']--;
            if (table[t.charAt(i) - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    // 只出现一次的数字三
    public int[] singleNumberThree(int[] nums) {
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        int lsb = (xor == Integer.MIN_VALUE ? xor : xor & (-xor));
        int t1 = 0, t2 = 0;
        for (int num : nums) {
            if ((num & lsb) != 0) {
                t1 ^= num;
            } else {
                t2 ^= num;
            }
        }
        return new int[]{t1, t2};
    }

    // 执行K此操作后的最大分数
    public long maxKelements(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> (b - a));
        for (int num : nums) {
            pq.add(num);
        }
        long ans = 0;
        for (int i = 0; i < k; i++) {
            int x = pq.poll();
            ans += x;
            pq.add((x + 2) / 3);
        }
        return ans;
    }

    // 同积元组
    public int tupleSameProduct(int[] nums) {
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                int x = nums[i] * nums[j];
                cnt.merge(x, 1, Integer::sum);
            }
        }
        int ans = 0;
        for (int v : cnt.values()) {
            ans += v * (v - 1) / 2;
        }
        return ans << 3;
    }

    // 根据规则将箱子分类
    public String categorizeBox(int length, int width, int height, int mass) {
        long v = (long) length * width * height;
        int bulky = length >= 10000 || width >= 10000 || height >= 10000 || v >= 1000000000 ? 1 : 0;
        int heavy = mass >= 100 ? 1 : 0;
        String[] d = {"Neither", "Bulky", "Heavy", "Both"};
        int i = heavy << 1 | bulky;
        return d[i];
    }

    // 老人的数目
    public int countSeniors(String[] details) {
        int ans = 0;
        for (String str : details) {
            if (Integer.parseInt(str.substring(11, 13)) > 60) {
                ans++;
            }
        }
        return ans;
    }

    // 掷骰子等于目标和的方法
    private final static int MOD = (int) (1e9 + 7);

    public int numRollsToTarget(int n, int k, int target) {
        int[][] f = new int[n + 1][target + 1];
        f[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= target; j++) {
                for (int m = 1; m <= k; m++) {
                    if (j >= m) {
                        f[i][j] = (f[i][j] + f[i - 1][j - m]) % MOD;
                    }
                }
            }
        }
        return f[n][target];
    }

    // 求一个整数的惩罚数
    public int punishmentNumber(int n) {
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            if (check(i * i, i)) {
                ans += i * i;
            }
        }
        return ans;
    }

    boolean check(int c, int t) {
        if (c == t) {
            return true;
        }
        int d = 10;
        while (c >= d && c % d <= t) {
            if (check(c / d, t - (c % d))) {
                return true;
            }
            d *= 10;
        }
        return false;
    }

    // 统计能整除数字的位数
    public int countDigits(int num) {
        int n = num, ans = 0;
        while (num != 0) {
            ans += n % (num % 10) == 0 ? 1 : 0;
            num /= 10;
        }
        return ans;
    }
}



