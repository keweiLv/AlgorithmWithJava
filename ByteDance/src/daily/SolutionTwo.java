package daily;

import java.util.*;

/**
 * @author Kezi
 * @date 2022年07月23日 22:39
 */
public class SolutionTwo {

    // 公交站台的距离
    public int distanceBetweenBusStops(int[] dist, int s, int t) {
        int n = dist.length, i = s, j = s, a = 0, b = 0;
        while (i != t) {
            a += dist[i];
            if (++i == n) {
                i = 0;
            }
        }
        while (j != t) {
            if (--j < 0) {
                j = n - 1;
            }
            b += dist[t];
        }
        return Math.min(a, b);
    }

    // 出现频率最高的K个数字
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        // 使用优先队列
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int num = entry.getKey(), cnt = entry.getValue();
            pq.offer(new int[]{num, cnt});
            if (pq.size() > k) {
                pq.poll();
            }
        }
        int[] ans = new int[pq.size()];
        for (int i = 0; i < k; i++) {
            ans[i] = pq.poll()[0];
        }
        return ans;
    }

    // 山峰数组额顶部
    public int peakIndexInMountainArray(int[] arr) {
        int n = arr.length;
        int left = 1, right = n - 2, ans = 0;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (arr[mid] > arr[mid + 1]) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }

    // 排序数组中只出现一次的数字
    public int singleNonDuplicate(int[] nums) {
        int n = nums.length, l = 0, r = n - 1;
        int ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (mid < n - 1 && nums[mid] == nums[mid + 1]) {
                if (mid % 2 == 0) {
                    l = mid + 2;
                } else {
                    r = mid - 1;
                }
            } else if (mid > 0 && nums[mid] == nums[mid - 1]) {
                if (mid % 2 == 0) {
                    r = mid - 2;
                } else {
                    l = mid + 1;
                }
            } else {
                ans = nums[mid];
                break;
            }
        }
        return ans;
    }

    /**
     * 有效的正方形
     * 该图形是正方形，那么任意三点组成的一定是等腰直角三角形，用此条件作为判断
     */
    long len = -1;

    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        return calc(p1, p2, p3) && calc(p1, p2, p4) && calc(p1, p3, p4) && calc(p2, p3, p4);
    }

    boolean calc(int[] a, int[] b, int[] c) {
        long l1 = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
        long l2 = (a[0] - c[0]) * (a[0] - c[0]) + (a[1] - c[1]) * (a[1] - c[1]);
        long l3 = (b[0] - c[0]) * (b[0] - c[0]) + (b[1] - c[1]) * (b[1] - c[1]);
        boolean ok = (l1 == l2 && l1 + l2 == l3) || (l1 == l3 && l1 + l3 == l2) || (l2 == l3 && l2 + l3 == l1);
        if (!ok) {
            return false;
        }
        if (len == -1) {
            len = Math.min(l1, l2);
        } else if (len == 0 || len != Math.min(l1, l2)) {
            return false;
        }
        return true;
    }

    // 链表的中间节点
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    // 环形链表Ⅱ
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head, slow = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return null;
            }
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                break;
            }
        }
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }

    // 层内元素之和
    public int maxLevelSum(TreeNode root) {
        int ans = 1, maxSum = root.val;
        List<TreeNode> q = new ArrayList<>();
        q.add(root);
        for (int lev = 1; !q.isEmpty(); ++lev) {
            List<TreeNode> nq = new ArrayList<>();
            int sum = 0;
            for (TreeNode node : q) {
                sum += node.val;
                if (node.left != null) {
                    nq.add(node.left);
                }
                if (node.right != null) {
                    nq.add(node.right);
                }
            }
            if (sum > maxSum) {
                maxSum = sum;
                ans = lev;
            }
            q = nq;
        }
        return ans;
    }

    // 生成每种字符串都是奇数个的字符串
    public String generateTheString(int n) {
        StringBuilder sb = new StringBuilder();
        if (n % 2 == 0 && --n >= 0) {
            sb.append("a");
        }
        while (n-- > 0) {
            sb.append("b");
        }
        return sb.toString();
    }

    // 分割字符串的最大得分
    public int maxScore(String s) {
        int n = s.length(), cur = s.charAt(0) == '0' ? 1 : 0;
        for (int i = 1; i < n; i++) {
            cur += s.charAt(i) - '0';
        }
        int ans = cur;
        for (int i = 1; i < n - 1; i++) {
            cur += s.charAt(i) == '0' ? 1 : -1;
            ans = Math.max(ans, cur);
        }
        return ans;
    }

    // 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }


    // 回文数字
    public boolean isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        }
        int rev = 0;
        while (x > rev) {
            rev = rev * 10 + x % 10;
            x /= 10;
        }
        return x == rev || x == rev / 10;
    }

    /**
     * 分割平衡字符串
     * 更好的方式是转换为数学判定，使用 1 来代指 L 得分，使用 -1 来代指 R 得分
     * 题目要求分割的 LR 子串尽可能多，直观上应该是尽可能让每个分割串尽可能短(归纳法证明该猜想的正确性)
     */
    public int balancedStringSplit(String s) {
        char[] cs = s.toCharArray();
        int n = cs.length;
        int ans = 0;
        for (int i = 0; i < n; ) {
            int j = i + 1, score = cs[i] == 'L' ? 1 : -1;
            while (j < n && score != 0) {
                score += cs[j++] == 'L' ? 1 : -1;
            }
            i = j;
            ans++;
        }
        return ans;
    }

    // 数组中第K大的元素
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(k, Comparator.comparingInt(a -> a));
        for (int num : nums) {
            pq.offer(num);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        return pq.peek();
    }

    // 一年中的第几天
    static int[] nums = new int[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    static int[] f = new int[13];

    static {
        for (int i = 1; i <= 12; i++) {
            f[i] = f[i - 1] + nums[i - 1];
        }
    }

    public int dayOfYear(String date) {
        String[] ss = date.split("-");
        int y = Integer.parseInt(ss[0]), m = Integer.parseInt(ss[1]), d = Integer.parseInt(ss[2]);
        boolean isLeap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        int ans = m > 2 && isLeap ? f[m - 1] + 1 : f[m - 1];
        return ans + d;
    }

    // 含有k个元素的组合
    List<List<Integer>> ans;

    public List<List<Integer>> combine(int n, int k) {
        ans = new ArrayList<>();
        dfs(n, 1, k, new ArrayList<>());
        return ans;
    }

    private void dfs(int n, int i, int k, List<Integer> list) {
        // 剪枝
        if (n - i + 1 < k) {
            return;
        }
        if (k == 0) {
            ans.add(new ArrayList<>(list));
            return;
        }
        list.add(i);
        dfs(n, i + 1, k - 1, list);
        list.remove(list.size() - 1);
        dfs(n, i + 1, k, list);
    }

    // 找到K个最接近的元素
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int n = arr.length, l = 0, r = n - 1;
        while (l < r) {
            int mid = l + r + 1 >> 1;
            if (arr[mid] <= x) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        r = r + 1 < n && Math.abs(arr[r + 1] - x) < Math.abs(arr[r] - x) ? r + 1 : r;
        int i = r - 1, j = r + 1;
        while (j - i - 1 < k) {
            if (i >= 0 && j < n) {
                if (Math.abs(arr[j] - x) < Math.abs(arr[i] - x)) {
                    j++;
                } else {
                    i--;
                }
            } else if (i >= 0) {
                i--;
            } else {
                j++;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (int p = i + 1; p <= j - 1; p++) {
            ans.add(arr[p]);
        }
        return ans;
    }

    // 重新排列数组
    public int[] shuffle(int[] nums, int n) {
        int[] ans = new int[2 * n];
        for (int i = 0, j = n, k = 0; k < 2 * n; n++) {
            ans[k] = k % 2 == 0 ? nums[i++] : nums[j++];
        }
        return ans;
    }

    // 最大二叉树Ⅱ
    // 懂了又好像没懂
    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        TreeNode node = new TreeNode(val);
        TreeNode prev = null, cur = root;
        while (cur != null && cur.val > val) {
            prev = cur;
            cur = cur.right;
        }
        if (prev == null) {
            node.left = cur;
            return node;
        } else {
            prev.right = node;
            node.left = cur;
            return root;
        }
    }

    // 验证栈序列
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> stack = new ArrayDeque<>();
        int n = pushed.length;
        for (int i = 0, j = 0; i < n; i++) {
            stack.push(pushed[i]);
            while (!stack.isEmpty() && stack.peek() == popped[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }

    // 商品折扣后的最终价格--单调栈
    public int[] finalPrices(int[] ps) {
        int n = ps.length;
        int[] ans = new int[n];
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!deque.isEmpty() && ps[deque.peekLast()] >= ps[i]) {
                int idx = deque.pollLast();
                ans[idx] = ps[idx] - ps[i];
            }
            deque.addLast(i);
            ans[i] = ps[i];
        }
        return ans;
    }

    // 最长同值路径
    int result = 0;

    public int longestUnivaluePath(TreeNode root) {
        calculate(root);
        return result;
    }

    public int calculate(TreeNode node) {
        if (node == null) return 0;
        int leftValue = calculate(node.left);
        int rightValue = calculate(node.right);
        leftValue = (node.left != null && node.val == node.left.val) ? ++leftValue : 0;
        rightValue = (node.right != null && node.val == node.right.val) ? ++rightValue : 0;
        result = Math.max(result, leftValue + rightValue);
        return Math.max(leftValue, rightValue);
    }

    // 二进制矩阵中的特殊位置
    public int numSpecial(int[][] mat) {
        int n = mat.length, m = mat[0].length, ans = 0;
        int[] r = new int[n], c = new int[m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
				r[i] += mat[i][j];c[j] += mat[i][j];
            }
        }
        for (int i = 0;i<n;i++){
            for (int j = 0;j<m;j++){
                if (mat[i][j] == 1 && r[i] == 1 && c[j] == 1){
                    ans++;
                }
            }
        }
        return ans;
    }
}