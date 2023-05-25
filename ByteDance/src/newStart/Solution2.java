package newStart;

import java.util.*;

public class Solution2 {

    // 两个非重叠子数组的最大和
    public int maxSumTwoNoOverlap(int[] nums, int firstLen, int secondLen) {
        int n = nums.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i++) {
            s[i + 1] = s[i] + nums[i];
        }
        return Math.max(f(s, firstLen, secondLen), f(s, secondLen, firstLen));
    }

    private int f(int[] s, int first, int sec) {
        int maxSumA = 0, res = 0;
        for (int i = first + sec; i < s.length; i++) {
            maxSumA = Math.max(maxSumA, s[i - sec] - s[i - sec - first]);
            res = Math.max(res, maxSumA + s[i] - s[i - sec]);
        }
        return res;
    }

    // 因子的组合
    public List<List<Integer>> getFactors(int n) {
        return dfs(2, n);
    }

    private List<List<Integer>> dfs(int start, int num) {
        if (num == 1) {
            return new ArrayList<>();
        }
        int qNum = (int) Math.sqrt(num);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = start; i <= qNum; i++) {
            if (num % i == 0) {
                List<Integer> simple = new ArrayList<>();
                simple.add(i);
                simple.add(num / i);
                result.add(simple);
                List<List<Integer>> nexList = dfs(i, num / i);
                for (List<Integer> list : nexList) {
                    list.add(i);
                    result.add(list);
                }
            }
        }
        return result;
    }

    // 最长字符串链
    Map<String, Integer> ws = new HashMap<>();

    public int longestStrChain(String[] words) {
        for (String str : words) {
            ws.put(str, 0);
        }
        int ans = 0;
        for (String key : ws.keySet()) {
            ans = Math.max(ans, dfs(key));
        }
        return ans;
    }

    private int dfs(String s) {
        Integer cnt = ws.get(s);
        if (cnt > 0) {
            return cnt;
        }
        for (int i = 0; i < s.length(); i++) {
            String tmp = s.substring(0, i) + s.substring(i + 1);
            if (ws.containsKey(tmp)) {
                cnt = Math.max(cnt, dfs(tmp));
            }
        }
        ws.put(s, cnt + 1);
        return cnt + 1;
    }

    // 寻找二叉树的叶子节点
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        while (root != null) {
            List<Integer> list = new ArrayList<>();
            root = recur(root, list);
            ans.add(list);
        }
        return ans;
    }

    private TreeNode recur(TreeNode root, List<Integer> list) {
        if (root == null) {
            return null;
        }
        if (root.left == null && root.right == null) {
            list.add(root.val);
            return null;
        }
        root.left = recur(root.left, list);
        root.right = recur(root.right, list);
        return root;
    }

    // 检查一个数是否在数组中占绝大多数
    public boolean isMajorityElement(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left + n / 2 < n && nums[left + n / 2] == target;
    }

    // 删除字符使频率相同
    public boolean equalFrequency(String word) {
        int[] set = new int[26];
        for (int i = 0; i < word.length(); i++) {
            set[word.charAt(i) - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            if (set[i] > 0) {
                set[i]--;
                int mark = 0;
                boolean ok = true;
                for (int num : set) {
                    if (num == 0) {
                        continue;
                    }
                    if (mark > 0 && num != mark) {
                        ok = false;
                        break;
                    }
                    mark = num;
                }
                if (ok) {
                    return true;
                }
                set[i]++;
            }
        }
        return false;
    }

    // 移动石子直到连续
    public int[] numMovesStones(int a, int b, int c) {
        int[] tmp = new int[]{a, b, c};
        Arrays.sort(tmp);
        a = tmp[0];
        b = tmp[1];
        c = tmp[2];
        return new int[]{
                c - a == 2 ? 0 : b - a <= 2 || c - b <= 2 ? 1 : 2, c - a - 2
        };
    }

    // 通知所有员工所需的时间
    public int numOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        List<Integer> g[] = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (int i = 0; i < n; i++) {
            if (manager[i] >= 0) {
                g[manager[i]].add(i);
            }
        }
        return dfs(g, informTime, headID);
    }

    private int dfs(List<Integer>[] g, int[] informTime, int id) {
        int maxTime = 0;
        for (int num : g[id]) {
            maxTime = Math.max(maxTime, dfs(g, informTime, num));
        }
        return maxTime + informTime[id];
    }

    // 强整数
    public List<Integer> powerfulIntegers(int x, int y, int bound) {
        Set<Integer> set = new HashSet<>();
        for (int a = 1; a <= bound; a *= x) {
            for (int b = 1; a + b <= bound; b *= y) {
                set.add(a + b);
                if (y == 1) {
                    break;
                }
            }
            if (x == 1) {
                break;
            }
        }
        return new ArrayList<>(set);
    }

    // 检查替换后的词是否有效
    public static boolean isValid(String s) {
        String tmp = s;
        while (!tmp.isBlank()) {
            String replace = tmp.replace("abc", "");
            if (replace.equals(tmp)) {
                return false;
            }
            tmp = replace;
        }
        return true;
    }

    // 摘水果
    public int maxTotalFruits(int[][] fruits, int startPos, int k) {
        int left = lowerBound(fruits, startPos - k);
        int right = left, s = 0, n = fruits.length;
        for (; right < n && fruits[right][0] <= startPos; right++) {
            s += fruits[right][1];
        }
        int ans = s;
        for (; right < n && fruits[right][0] <= startPos + k; right++) {
            s += fruits[right][1];
            while (fruits[right][0] * 2 - fruits[left][0] - startPos > k && fruits[right][0] - fruits[left][0] * 2 + startPos > k) {
                s -= fruits[left++][1];
            }
            ans = Math.max(ans, s);
        }
        return ans;
    }

    private int lowerBound(int[][] fruits, int target) {
        int left = -1, right = fruits.length;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (fruits[mid][0] < target) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return right;
    }

    // 有序数组中的缺失元素
    public int missingElement(int[] nums, int k) {
        int lose = nums[nums.length - 1] - nums[0] + 1 - nums.length;
        if (k > lose) {
            return nums[nums.length - 1] + k - lose;
        }
        int l = 0;
        int r = nums.length - 1;
        while (l < r - 1) {
            int mid = l + (r - l) >> 1;
            int loseCnt = nums[mid] - nums[l] - (mid - l);
            if (k > loseCnt) {
                l = mid;
                k -= loseCnt;
            } else {
                r = mid;
            }
        }
        return nums[l] + k;
    }

    // 处理用时最长的那个任务的员工
    public int hardestWorker(int n, int[][] logs) {
        int ans = 0;
        int last = 0, mx = 0;
        for (int[] log : logs) {
            int uid = log[0], time = log[1];
            time -= last;
            if (time > mx || (time == mx && uid < ans)) {
                ans = uid;
                mx = time;
            }
            last += time;
        }
        return ans;
    }

    // 找出变味映射
    public int[] anagramMappings(int[] nums1, int[] nums2) {
        Map<Integer, Integer> D = new HashMap<>();
        for (int i = 0; i < nums2.length; i++) {
            D.put(nums2[i], i);
        }
        int[] ans = new int[nums1.length];
        int t = 0;
        for (int x : nums1) {
            ans[t++] = D.get(x);
        }
        return ans;
    }

    // 数青蛙
    private static final char[] pre = new char['s'];

    static {
        char[] charArray = "croakc".toCharArray();
        for (int i = 1; i < charArray.length; i++) {
            pre[charArray[i]] = charArray[i - 1];
        }
    }

    public int minNumberOfFrogs(String croakOfFrogs) {
        int[] cnt = new int['s'];
        for (char ch : croakOfFrogs.toCharArray()) {
            char c = pre[ch];
            if (cnt[c] > 0) {
                cnt[c]--;
            } else if (ch != 'c') {
                return -1;
            }
            cnt[ch]++;
        }
        if (cnt['c'] > 0 || cnt['r'] > 0 || cnt['o'] > 0 || cnt['a'] > 0) {
            return -1;
        }
        return cnt['k'];
    }

    // 单线程CPU
    public int[] getOrder(int[][] tasks) {
        int n = tasks.length;
        int[][] nt = new int[n][3];
        for (int i = 0; i < n; i++) {
            nt[i] = new int[]{tasks[i][0], tasks[i][1], i};
        }
        Arrays.sort(nt, (a, b) -> a[0] - b[0]);
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            if (a[1] != b[1]) {
                return a[1] - b[1];
            }
            return a[2] - b[2];
        });
        int[] ans = new int[n];
        for (int time = 1, j = 0, idx = 0; idx < n; ) {
            while (j < n && nt[j][0] <= time) {
                pq.add(nt[j++]);
            }
            if (pq.isEmpty()) {
                time = nt[j][0];
            } else {
                int[] poll = pq.poll();
                ans[idx++] = poll[2];
                time += poll[1];
            }
        }
        return ans;
    }

    // 总持续时间可被60整除的歌曲
    public int numPairsDivisibleBy60(int[] time) {
        int ans = 0;
        int[] cnt = new int[60];
        for (int t : time) {
            ans += cnt[(60 - t % 60) % 60];
            cnt[t % 60]++;
        }
        return ans;
    }

    // 雇佣K名工人的最低成本
    public double mincostToHireWorkers(int[] qs, int[] ws, int k) {
        int n = qs.length;
        double[][] ds = new double[n][2];
        for (int i = 0; i < n; i++) {
            ds[i][0] = ws[i] * 1.0 / qs[i];
            ds[i][1] = i * 1.0;
        }
        Arrays.sort(ds, (a, b) -> Double.compare(a[0], b[0]));
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        double ans = 1e18;
        for (int i = 0, tol = 0; i < n; i++) {
            int cur = qs[(int) ds[i][1]];
            tol += cur;
            pq.add(cur);
            if (pq.size() > k) {
                tol -= pq.poll();
            }
            if (pq.size() == k) {
                ans = Math.min(ans, tol * ds[i][0]);
            }
        }
        return ans;
    }

    // 有效时间的数目
    public int countTime(String time) {
        return func(time.substring(0, 2), 24) * func(time.substring(3, 5), 60);
    }

    private int func(String time, int m) {
        int cnt = 0;
        for (int i = 0; i < m; i++) {
            boolean a = time.charAt(0) == '?' || time.charAt(0) - '0' == i / 10;
            boolean b = time.charAt(1) == '?' || time.charAt(1) - '0' == i % 10;
            cnt += a && b ? 1 : 0;
        }
        return cnt;
    }

    // 可以被K整除的最小整数
    public int smallestRepunitDivByK(int k) {
        Set<Integer> set = new HashSet<>();
        int x = 1 % k;
        while (x > 0 && set.add(x)) {
            x = (x * 10 + 1) % k;
        }
        return x > 0 ? -1 : set.size() + 1;
    }

    // 子串能表示从1到N数字的二进制串
    public boolean queryString(String s, int n) {
        for (int i = 1; i <= n; i++) {
            String binaryString = Integer.toBinaryString(i);
            if (!s.contains(binaryString)) {
                return false;
            }
        }
        return true;
    }

    // 与对应负数同时存在的最大正整数
    public int findMaxK(int[] nums) {
        int ans = -1;
        Set<Integer> set = new HashSet<>();
        for (int x : nums) {
            set.add(x);
        }
        for (int nu : set) {
            if (set.contains(-nu)) {
                ans = Math.max(ans, nu);
            }
        }
        return ans;
    }

    // 所有子集
    private final List<List<Integer>> ans = new ArrayList<>();
    private final List<Integer> path = new ArrayList<>();
    private int[] nums;

    public List<List<Integer>> subsets(int[] nums) {
        this.nums = nums;
        dfs(0);
        return ans;
    }

    private void dfs(int i) {
        if (i == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }
        // 不选nums[i]
        dfs(i + 1);
        // 选nums[i]
        path.add(nums[i]);
        dfs(i + 1);
        path.remove(path.size() - 1);
    }

    // 距离相等的条形码
    public int[] rearrangeBarcodes(int[] barcodes) {
        int n = barcodes.length;
        Integer[] tmp = new Integer[n];
        int max = 0;
        for (int i = 0; i < n; i++) {
            tmp[i] = barcodes[i];
            max = Math.max(max, barcodes[i]);
        }
        int[] cnt = new int[max + 1];
        for (int x : barcodes) {
            cnt[x]++;
        }
        Arrays.sort(tmp, (a, b) -> cnt[a] == cnt[b] ? a - b : cnt[b] - cnt[a]);
        int[] ans = new int[n];
        for (int k = 0, j = 0; k < 2; k++) {
            for (int i = k; i < n; i += 2) {
                ans[i] = tmp[j++];
            }
        }
        return ans;
    }

    // 字符串轮转
    public boolean isFlipedString(String s1, String s2) {
        return s1.length() == s2.length() && (s1 + s1).contains(s2);
    }

    // 重复的DNA序列
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> ans = new ArrayList<>();
        int n = s.length();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i + 10 <= n; i++) {
            String cur = s.substring(i, i + 10);
            int cnt = map.getOrDefault(cur, 0);
            if (cnt == 1) {
                ans.add(cur);
            }
            map.put(cur, cnt + 1);
        }
        return ans;
    }

    // 按列翻转得到最大值等行数
    public int maxEqualRowsAfterFlips(int[][] matrix) {
        int ans = 0, n = matrix[0].length;
        Map<String, Integer> cnt = new HashMap<>(16);
        for (int[] row : matrix) {
            int[] r = new int[n];
            for (int i = 0; i < n; i++) {
                r[i] = (char) (row[0] ^ row[i]);
            }
            Integer merge = cnt.merge(Arrays.toString(r), 1, Integer::sum);
            ans = Math.max(ans, merge);
        }
        return ans;
    }

    // 工作计划的最低难度
    private int[] a;
    private int[][] memo;

    public int minDifficulty(int[] jobDifficulty, int d) {
        int n = jobDifficulty.length;
        if (n < d) {
            return -1;
        }
        this.a = jobDifficulty;
        memo = new int[d][n];
        for (int i = 0; i < d; i++) {
            Arrays.fill(memo[i], -1);
        }
        return jobDfs(d - 1, n - 1);
    }

    private int jobDfs(int d, int n) {
        if (memo[d][n] != -1) {
            return memo[d][n];
        }
        if (d == 0) {
            int mx = 0;
            for (int k = 0; k <= n; k++) {
                mx = Math.max(mx, a[k]);
            }
            return memo[d][n] = mx;
        }
        int res = Integer.MAX_VALUE;
        int mx = 0;
        for (int k = n; k >= d; k--) {
            mx = Math.max(mx, a[k]);
            res = Math.min(res, jobDfs(d - 1, k - 1) + mx);
        }
        return memo[d][n] = res;
    }

    // 文件的最长绝对路径
    static int[] hash = new int[10010];

    public int lengthLongestPath(String input) {
        Arrays.fill(hash, -1);
        int n = input.length(), ans = 0;
        for (int i = 0; i < n; ) {
            int level = 0;
            while (i < n && input.charAt(i) == '\t' && ++level >= 0) {
                i++;
            }
            int j = i;
            boolean isDir = true;
            while (j < n && input.charAt(j) != '\n') {
                if (input.charAt(j++) == '.') {
                    isDir = false;
                }
            }
            int cur = j - i;
            int pre = level - 1 >= 0 ? hash[level - 1] : -1;
            int path = pre + 1 + cur;
            if (isDir) {
                hash[level] = path;
            } else if (path > ans) {
                ans = path;
            }
            i = j + 1;
        }
        return ans;
    }

    /**
     * 最大BST子树
     * int[]数组，[0]是否是BST，[1]当前为跟的BST的最最小值，[2]最大值，[3]该BST节点树
     */

    int largestBST = 0;

    public int largestBSTSubtree(TreeNode root) {
        dfs(root);
        return largestBST;
    }

    private int[] dfs(TreeNode root) {
        if (root == null) {
            return new int[]{1, Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
        }
        int[] left = dfs(root.left);
        int[] right = dfs(root.right);
        int[] cur = new int[4];
        if (left[0] == 1 && right[0] == 1 && left[2] < root.val && right[1] > root.val) {
            cur[0] = 1;
            cur[1] = Math.min(left[1], root.val);
            cur[2] = Math.max(right[2], root.val);
            cur[3] = left[3] + right[3] + 1;
            largestBST = Math.max(largestBST, cur[3]);
        } else {
            cur[0] = 0;
        }
        return cur;
    }


    //判断两个事件是否存在冲突
    public boolean haveConflict(String[] event1, String[] event2) {
        return !(event1[0].compareTo(event2[1]) > 0 || event1[1].compareTo(event2[0]) < -0);
    }

    // 二叉树的垂直遍历
    public List<List<Integer>> verticalOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        Map<Integer, List<Integer>> res = new TreeMap<>();
        Map<TreeNode, Integer> nodeMap = new HashMap<>();
        nodeMap.put(root, 0);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode poll = queue.poll();
            int i = nodeMap.get(poll);
            res.computeIfAbsent(i, k -> new ArrayList<>()).add(poll.val);
            if (poll.left != null) {
                queue.add(poll.left);
                nodeMap.put(poll.left, i - 1);
            }
            if (poll.right != null) {
                queue.add(poll.right);
                nodeMap.put(poll.right, i + 1);
            }
        }
        return new ArrayList<>(res.values());
    }

    // 反转每对括号间的子串
    public String reverseParentheses(String s) {
        Deque<String> stack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                stack.push(sb.toString());
                sb.setLength(0);
            } else if (ch == ')') {
                sb.reverse();
                sb.insert(0, stack.pop());
            } else {
                sb.append(ch);
            }
        }
        return sb.toString();
    }

    // 查找两颗二叉搜索树之和
    public boolean twoSumBSTs(TreeNode root1, TreeNode root2, int target) {
        if (root1 == null || root2 == null) {
            return false;
        }
        List<TreeNode> listA = new ArrayList<>();
        List<TreeNode> listB = new ArrayList<>();
        inorder(root1, listA);
        inorder(root2, listB);
        int lenA = listA.size();
        int lenB = listB.size();
        int pa = 0, pb = lenB - 1;
        while (pa < lenA && pb >= 0) {
            int valA = listA.get(pa).val;
            int valB = listB.get(pb).val;
            if (valA + valB == target) {
                return true;
            } else if (valA + valB > target) {
                pb--;
            } else {
                pa++;
            }
        }
        return false;
    }

    private void inorder(TreeNode root, List<TreeNode> list) {
        if (root == null) {
            return;
        }
        inorder(root.left, list);
        list.add(root);
        inorder(root.right, list);
    }

    // 活字印刷
    public int numTilePossibilities(String tiles) {
        int[] cnt = new int[26];
        for (char c : tiles.toCharArray()) {
            ++cnt[c - 'A'];
        }
        return dfs(cnt);
    }

    private int dfs(int[] cnt) {
        int res = 0;
        for (int i = 0; i < cnt.length; i++) {
            if (cnt[i] > 0) {
                ++res;
                --cnt[i];
                res += dfs(cnt);
                ++cnt[i];
            }
        }
        return res;
    }

    // 打家劫舍
    public int rob(int[] nums) {
        int pre = 0;
        int cur = 0;
        for (int num : nums) {
            int temp = Math.max(pre + num, cur);
            pre = cur;
            cur = temp;
        }
        return cur;
    }

    // 二叉树中最长的连续序列
    int maxLen = 0;

    public int longestConsecutive(TreeNode root) {
        longestPath(root);
        return maxLen;
    }

    private int[] longestPath(TreeNode root) {
        if (root == null) {
            return new int[]{0, 0};
        }
        int inr = 1, dcr = 1;
        if (root.left != null) {
            int[] l = longestPath(root.left);
            if (root.val == root.left.val - 1) {
                inr = l[0] + 1;
            } else if (root.val == root.left.val + 1) {
                dcr = l[1] + 1;
            }
        }
        if (root.right != null) {
            int[] r = longestPath(root.right);
            if (root.val == root.right.val + 1) {
                dcr = Math.max(dcr, r[1] + 1);
            } else if (root.val == root.right.val - 1) {
                inr = Math.max(inr, r[0] + 1);
            }
        }
        maxLen = Math.max(maxLen, dcr + inr - 1);
        return new int[]{inr, dcr};
    }

    // 至多包含k个不同字符的最长子串
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        int n = s.length();
        if (n * k == 0) {
            return 0;
        }
        if (k > n) {
            return k;
        }
        int left = 0, right = 0;
        Map<Character, Integer> map = new HashMap<>();
        int ans = 1;
        while (right < n) {
            map.put(s.charAt(right), right++);
            if (map.size() > k) {
                Integer min = Collections.min(map.values());
                map.remove(s.charAt(min));
                left = min + 1;
            }
            ans = Math.max(ans, right - left);
        }
        return ans;
    }

    // 二叉搜索子树的最大键值和
    int maxSum = 0;

    public int maxSumBST(TreeNode root) {
        maxSumDfs(root);
        return maxSum;
    }

    private int[] maxSumDfs(TreeNode root) {
        if (root == null) {
            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
        }
        int[] left = maxSumDfs(root.left);
        int[] right = maxSumDfs(root.right);
        int cur = root.val;
        if (cur <= left[1] || cur >= right[0]) {
            return new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE, 0};
        }
        int s = left[2] + right[2] + cur;
        maxSum = Math.max(maxSum, s);
        return new int[]{Math.min(left[0], cur), Math.max(right[1], cur), s};
    }

    // 滑动窗口最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            return a[0] != b[0] ? b[0] - a[0] : b[1] - a[1];
        });
        for (int i = 0; i < k; i++) {
            pq.add(new int[]{nums[i], i});
        }
        int[] ans = new int[n - k + 1];
        ans[0] = pq.peek()[0];
        for (int i = k; i < n; i++) {
            pq.add(new int[]{nums[i], i});
            while (pq.peek()[1] <= i - k) {
                pq.poll();
            }
            ans[i - k + 1] = pq.peek()[0];
        }
        return ans;
    }

    // 柱状图中最大的矩形
    public int largestRectangleArea(int[] hs) {
        int n = hs.length;
        int[] l = new int[n], r = new int[n];
        Arrays.fill(l, -1);
        Arrays.fill(r, n);
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!deque.isEmpty() && hs[deque.peekLast()] > hs[i]) {
                r[deque.pollLast()] = i;
            }
            deque.addLast(i);
        }
        deque.clear();
        for (int i = n - 1; i >= 0; i--) {
            while (!deque.isEmpty() && hs[deque.peekLast()] > hs[i]) {
                l[deque.pollLast()] = i;
            }
            deque.addLast(i);
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int h = hs[i], a = l[i], b = r[i];
            ans = Math.max(ans, h * (a - b - 1));
        }
        return ans;
    }

    // 蓄水
    public int storeWater(int[] bucket, int[] vat) {
        int max = Arrays.stream(vat).max().getAsInt();
        if (max == 0) {
            return 0;
        }
        int ans = Integer.MAX_VALUE;
        int n = bucket.length;
        for (int i = 1; i <= max; i++) {
            int y = 0;
            for (int j = 0; j < n; j++) {
                y += Math.max(0, (vat[j] + i - 1) / i - bucket[j]);
            }
            ans = Math.min(ans, i + y);
        }
        return ans;
    }

    // 子数组范围和
    public long subArrayRanges(int[] nums) {
        int n = nums.length;
        long ans = 0;
        for (int i = 0; i < n; i++) {
            int min = nums[i], max = nums[i];
            for (int j = i + 1; j < n; j++) {
                min = Math.min(min, nums[j]);
                max = Math.max(max, nums[j]);
                ans += max - min;
            }
        }
        return ans;
    }

    // 区间子数组个数
    public int numSubarrayBoundedMax(int[] nums, int left, int right) {
        int n = nums.length, ans = 0;
        for (int i = 0, j = -1, k = -1; i < n; i++) {
            if (nums[i] > right) {
                k = i;
            } else {
                if (nums[i] < left) {
                    if (j > k) {
                        ans += j - k;
                    }
                } else {
                    ans += i - k;
                    j = i;
                }
            }
        }
        return ans;
    }

    // 最大连续1的个数二
    public int findMaxConsecutiveOnes(int[] nums) {
        int res = 0, cnt = 0;
        for (int l = 0, r = 0; r < nums.length; r++) {
            if (nums[r] == 0) {
                cnt++;
                while (cnt > 1) {
                    cnt -= nums[l++] == 0 ? 1 : 0;
                }
            }
            res = Math.max(res, r - l + 1);
        }
        return res;
    }

    // 根到叶路径上的不足节点
    public TreeNode sufficientSubset(TreeNode root, int limit) {
        limit -= root.val;
        if (root.right == null && root.left == null) {
            return limit > 0 ? null : root;
        }
        if (root.left != null) {
            root.left = sufficientSubset(root.left, limit);
        }
        if (root.right != null) {
            root.right = sufficientSubset(root.right, limit);
        }
        return root.left == null && root.right == null ? null : root;
    }

    // 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] ans = new int[n];
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (deque.peekLast() != null && temperatures[deque.peekLast()] < temperatures[i]) {
                int idx = deque.pollLast();
                ans[idx] = i - idx;
            }
            deque.offerLast(i);
        }
        return ans;
    }

    // 回文排列
    public boolean canPermutePalindrome(String s) {
        Set<Character> set = new HashSet<>();
        for (int i = 0; i < s.length(); i++) {
            if (!set.add(s.charAt(i))) {
                set.remove(s.charAt(i));
            }
        }
        return set.size() <= 1;
    }

    // 受标签影响的最大值
    public int largestValsFromLabels(int[] values, int[] labels, int numWanted, int useLimit) {
        int n = values.length;
        Integer[] tmp = new Integer[n];
        for (int i = 0; i < n; i++) {
            tmp[i] = i;
        }
        Arrays.sort(tmp, (a, b) -> values[b] - values[a]);
        int ans = 0, choose = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n && choose < numWanted; i++) {
            int label = labels[tmp[i]];
            if (map.getOrDefault(label, 0) == useLimit) {
                continue;
            }
            choose++;
            ans += values[tmp[i]];
            map.put(label, map.getOrDefault(label, 0) + 1);
        }
        return ans;
    }

    // 单词替换-字段树
    class Node {
        boolean isend;
        Node[] tns = new Node[26];
    }

    Node root = new Node();

    void add(String s) {
        Node p = root;
        for (int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) {
                p.tns[u] = new Node();
            }
            p = p.tns[u];
        }
        p.isend = true;
    }

    String query(String s) {
        Node p = root;
        for (int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) {
                break;
            }
            if (p.tns[u].isend) {
                return s.substring(0, i + 1);
            }
            p = p.tns[u];
        }
        return s;
    }

    public String replaceWords(List<String> dictionary, String sentence) {
        for (String str : dictionary) {
            add(str);
        }
        StringBuilder sb = new StringBuilder();
        for (String str : sentence.split(" ")) {
            sb.append(query(str)).append(" ");
        }
        return sb.substring(0, sb.length() - 1);
    }

    // 单行键盘
    public int calculateTime(String keyboard, String word) {
        int ans = 0, start = 0;
        int[] rec = new int[26];
        for (int i = 0; i < keyboard.length(); i++) {
            rec[keyboard.charAt(i) - 'a'] = i;
        }
        for (char c : word.toCharArray()) {
            ans += Math.abs((rec[c - 'a']) - start);
            start = rec[c - 'a'];
        }
        return ans;
    }

    // 缺失的区间
    public List<List<Integer>> findMissingRanges(int[] nums, int lower, int upper) {
        List<List<Integer>> res = new ArrayList<>();
        long pre = lower - 1;
        for (int i = 0; i < nums.length; i++) {
            List<Integer> tmp = new ArrayList<>();
            if (nums[i] - pre >= 2) {
                tmp.add((int) (pre + 1));
                tmp.add(nums[i] - 1);
            }
            pre = nums[i];
            if (tmp.size() > 0) {
                res.add(tmp);
            }
        }
        if (upper - pre >= 1) {
            List<Integer> tmp = new ArrayList<>();
            tmp.add((int) (pre + 1));
            tmp.add(upper);
            res.add(tmp);
        }
        return res;
    }

    // 差值数组不同的字符串
    public String oddString(String[] words) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : words) {
            int n = str.length();
            char[] cs = new char[n - 1];
            for (int i = 0; i < n - 1; i++) {
                cs[i] = (char) (str.charAt(i + 1) - str.charAt(i));
            }
            String s = String.valueOf(cs);
            map.computeIfAbsent(s, k -> new ArrayList<>()).add(str);
        }
        for (List<String> it : map.values()) {
            if (it.size() == 1) {
                return it.get(0);
            }
        }
        return "";
    }

    // 删掉链表M个节点之后的N个节点
    public ListNode deleteNodes(ListNode head, int m, int n) {
        ListNode cur = head;
        while (cur != null){
            int num = 1;
            while (num < m && cur != null){
                cur = cur.next;
                num++;
            }
            if (cur == null){
                return head;
            }
            num = 0;
            while (num < n && cur.next != null) {
                cur.next = cur.next.next;
                num++;
            }
            cur = cur.next;
        }
        return head;
    }
}
