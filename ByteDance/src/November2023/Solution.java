package November2023;

import java.util.*;
import java.util.stream.IntStream;

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

    // 数组中两个数的最大异或值
    public int findMaximumXOR(int[] nums) {
        int res = 0;
        int mask = 0;
        for (int i = 30; i >= 0; i--) {
            mask = mask | (1 << i);

            Set<Integer> set = new HashSet<>();
            for (int num : nums) {
                set.add(num & mask);
            }
            int temp = res | (1 << i);
            for (Integer prefix : set) {
                if (set.contains(prefix ^ temp)) {
                    res = temp;
                    break;
                }
            }
        }
        return res;
    }

    // 重复的DNA序列
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> res = new ArrayList<>();
        int n = s.length();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i + 10 <= n; i++) {
            String cur = s.substring(i, i + 10);
            int cnt = map.getOrDefault(cur, 0);
            if (cnt == 1) {
                res.add(cur);
            }
            map.put(cur, cnt + 1);
        }
        return res;
    }

    // 最大单词长度乘积
    public int maxProduct(String[] words) {
        Map<Integer, Integer> map = new HashMap<>();
        for (String word : words) {
            int t = 0, n = word.length();
            for (int i = 0; i < n; i++) {
                int c = word.charAt(i) - 'a';
                t |= (1 << c);
            }
            if (!map.containsKey(t) || map.get(t) < n) {
                map.put(t, n);
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

    // 优势洗牌
    public int[] advantageCount(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int[] ans = new int[n];
        Arrays.sort(nums1);
        Integer[] array = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(array, (i, j) -> nums2[i] - nums2[j]);
        int left = 0, right = n - 1;
        for (int num : nums1) {
            ans[num > nums2[array[left]] ? array[left++] : array[right--]] = num;
        }
        return ans;
    }

    // 盛最多水的容器
    public int maxArea(int[] height) {
        int i = 0, j = height.length - 1, res = 0;
        while (i < j) {
            res = height[i] < height[j] ?
                    Math.max(res, (j - i) * height[i++]) :
                    Math.max(res, (j - i) * height[j--]);
        }
        return res;
    }
}
