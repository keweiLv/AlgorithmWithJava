package November2023;

import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {

    // 水果成篮
    public int totalFruit(int[] fruits) {
        int n = fruits.length, ans = 0;
        int[] cnts = new int[n + 10];
        for (int i = 0, j = 0, tol = 0; i < n; i++) {
            if (++cnts[fruits[i]] == 1) {
                tol++;
            }
            while (tol > 2) {
                if (--cnts[fruits[j++]] == 0) {
                    tol--;
                }
            }
            ans = Math.max(ans, i - j + 1);
        }
        return ans;
    }

    // K个不同整数的子数组
    public int subarraysWithKDistinct(int[] nums, int k) {
        int len = nums.length;
        int ans = 0;
        int count = 0;
        int left = 0;
        int right = 0;
        int[] map = new int[len + 1];
        while (right < len) {
            if (map[nums[right]] == 0) {
                count++;
            }
            map[nums[right]]++;
            right++;
            while (count > k) {
                map[nums[left]]--;
                if (map[nums[left]] == 0) {
                    count--;
                }
                left++;
            }
            ans += right - left;
        }
        return ans;
    }

    // 填充每个节点的下一个右侧节点指针二
    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    ;

    public Node connect(Node root) {
        Node res = root;
        if (root == null) {
            return res;
        }
        Deque<Node> deque = new ArrayDeque<>();
        deque.addLast(root);
        while (!deque.isEmpty()) {
            int sz = deque.size();
            while (sz-- > 0) {
                Node cur = deque.pollFirst();
                if (sz != 0) {
                    cur.next = deque.peekFirst();
                }
                if (cur.left != null) {
                    deque.addLast(cur.left);
                }
                if (cur.right != null) {
                    deque.addLast(cur.right);
                }
            }
        }
        return res;
    }

    // 分隔链表
    public ListNode[] splitListToParts(ListNode head, int k) {
        int cnt = 0;
        ListNode temp = head;
        while (temp != null && ++cnt > 0) {
            temp = temp.next;
        }
        int pre = cnt / k;
        ListNode[] ans = new ListNode[pre];
        for (int i = 0, sum = 1; i < k; i++, sum++) {
            ans[i] = head;
            temp = ans[i];
            int u = pre;
            while (u-- > 0 && ++sum > 0) {
                temp = temp.next;
            }
            int remain = k - i - 1;
            if (pre != 0 && sum + pre * remain < cnt && ++sum > 0) {
                temp = temp.next;
            }
            head = temp != null ? temp.next : null;
            if (temp != null) {
                temp.next = null;
            }
        }
        return ans;
    }
}
