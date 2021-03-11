# 剑指offer算法记录

### 1. **二维数组中的查找**

> *在一个二维数组中（每个一维数组的长度相同），每一行都按照**从左到右递增**的顺序排序，每一列都按照**从上到下递增**的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。*

解题思路：

1. “递增”联想到二分法。暴力算法是全部循环遍历比较。
2. 巧妙设初始值value为右上角/左下角的数。以右上角为例。
3. 如果value==target，元素找到，直接返回。
4. 如果value<target，也就是说target大于这一行的所有值，所以将value向下移一行。
5. 如果value<target，也就是说target小于这一列的所有值，所以将value向左移一行。
6. 重复以上步骤。

算法复杂度：

+ 时间复杂度： O(m+n)， m行，n列，最坏情况遍历m+n次。  

+ 空间复杂度：O(1)

代码：

```python
def Find(self, target, array):    
    if not array:
        return False
    rows = len(array) #行
    columns = len(array[0]) #列
    row = 0
    column = columns - 1 #从0开始的
    while column >= 0 and row < rows:
        if target == array[row][column]:
            return True
        if target < array[row][column]:
            column -= 1
        else:
            row += 1
    return False
```
### 2. **替换空格**

> *请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。*

解题思路：

1. 想法一：直接使用python内置函数replace

2. 想法二：遍历字符串，每次遇到空格就在新字符串后加上%20

3. > 想法三：逆向遍历（C）
   >
   > 1. in-place-返回值为void，说明不能另外开辟数组。从后往前插入%20。
   > 2. 如果从左到右，会发现如果遇到空格，会将原来的字符覆盖。于是，此方法不行。
   > 3. 考虑从右向左，遇到空格，就填充“20%“，否则将原字符移动应该呆的位置。
   > 4. length为原字符串最后一个字符的位置，new_length为结果字符串的最后一个位置。
   > 5. 如果str[length]不等于空格，就复制，然后指针分别左移一位。
   > 6. 如果str[length]等于空格，就填充“20%”。
   > 7. 一直重复上述步骤，直到字符串遍历完毕。

代码：

```python
def replaceSpace(self, s):
	return s.replace(' ', '%20')
```

```python
def replaceSpace(self, s):
	new_s = ''
	for i in s:
		if i == ' ':
			new_s += '%20'
		else:
			new_s += i
	return new_s
```

### 3. **从尾到头打印链表**

> *输入一个链表，按链表从尾到头的顺序返回一个ArrayList。*

解题思路：

1. 想法一：

   > 1. 使用栈（先进后出）。先判断链表是否为空，若是则返回空值。
   > 2. 进栈：遍历listNode并将value拼接，语法是listNode.val，每拼接一个就获取下一个节点，语法是listNode.next。temp.append(listNode.val)。
   > 3. 出栈：循环进栈中拼接完成的temp，最后结果result是拼接（append）temp依序弹出相应值，语法是temp.pop()，返回result

2. 想法二：

   > 1. 从头到尾遍历，逆序输出。
   > 2. 遍历的方法与想法一的进栈相同。
   > 3. 逆序输出即使用list[::-1]

代码：

1. ```python
   def printListFromTailToHead(self, listNode):
           # write code here
           if not listNode:
               return []
           temp = []
           result = []
           while listNode:
               temp.append(listNode.val)
               listNode = listNode.next
           while temp:
               result.append(temp.pop())
           return result
   ```

2. ```python
   def printListFromTailToHead(self, listNode):
           if not listNode:
               return []
           result = []
           while listNode:
               result.append(listNode.val)
               listNode = listNode.next
           return result[::-1]
   ```

### 4. **重建二叉树**

> *输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。*

涉及考点：二叉树

**先序**

> if 二叉树为空 - 空操作
>
> if 不为空：
>
> 	1. 访问根节点
>  2. 先序遍历左子树
>  3. 先序遍历右子树

```python
def preorder(self):
	if self.data is not None:
		print(self.data, end='')
    if self.left is not None:
    	self.left.preorder()
    if self.right is not None:
    	self.right.preorder()
```

**中序**

> if 二叉树为空 - 空操作
>
> if 不为空：
>
>  1. 中序遍历左子树
>  2. 访问根节点
>  3. 中序遍历右子树

```python
def inorder(self):
    if self.left is not None:
        self.left.inorder()
    if self.data is not None:
        print(self.data, end='')
    if self.right is not None:
        self.right.inorder()
```

**后序**

> if 二叉树为空 - 空操作
>
> if 不为空：
>
>  1. 后序遍历左子树
>  2. 后序遍历右子树
>  3. 访问根节点

``` python 
def postorder(self):
    if self.left is not None:
        self.left.postorder()
    if self.right is not None:
        self.right.postorder()
    if self.data is not None:
        print(self.data, end='')
```

**层序**

> if 二叉树为空
>
> if 非空
>
> ​	从上向下，从左向右
>
> 按层次进行访问

解题思路：

1. 首先判断是否为空数，若是则返回None

2. 判断是否只有一个节点，若是则直接返回TreeNode(pre[0])，直接构造TreeNode，根节点为pre[0]

3. 题中给出了前序和中序遍历，可以得出的结论是：前序遍历的第一个数是二叉树的根节点pre[0]。在中序遍历中，根节点之前的是左子树，之后的为右子树。

4. 构建递归函数reConstructBinaryTree(self, pre, tin)。

5. 每一个根节点左子树右子树的组合都是遵循前序和中序遍历的规则的，所以使用递归的思想。

   1. > 左子树的个数：中序遍历中根节点之前的所有数字。
      >
      > 根节点为值为pre[0]的数，通过tin.index(pre[0])来获取index的值。比如index为2则说明左子树一共有2个数字（从0开始）。
      >
      > 在前序遍历中，因为左子树是从前序遍历的第二个数字开始的，所以第一个数字index为1，最后一个数字index是1+tin.index(pre[0])，所以左子树的前序遍历为pre[1:tin.index(pre[0])+1]
      >
      > 在中序遍历中，左子树为根节点之前的所有数，即tin[:tin.index(pre[0])] 

   2. > 右子树的个数：中序遍历中根节点之后的所有数字。
      >
      > 在前序遍历中，右子树为左子树之后的所有数，所以右子树的前序遍历为pre[tin.index(pre[0])+1:]
      >
      > 在中序遍历中，右子树为根节点之后的所有数，所以中序遍历为tin[tin.index(pre[0])+1:]

代码：

``` python
def reConstructBinaryTree(self, pre, tin):
    if len(pre) == 0:
        return None
    elif len(pre) == 1:
        return TreeNode(pre[0])
    elif:
        ans = TreeNode(pre[0]) //定义根节点并构造空树
        ans.left = self.reConstructBinaryTree(pre[1: tin.index(pre[0])+1], tin[:tin.index(pre[0])])
        ans.right = self.reConstructBinaryTree(pre[tin.index(pre[0]+1):], tin[tin.index(pre[0])+1:])
        return ans
```



### 5. 用两个栈实现队列

> *用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。*

涉及考点：

队列：FIFO	栈：FILO

解题思路：

1. 进队列的过程中只需要有容器来装元素就可以了，所以直接往栈1里面压数据。使用append拼接即可。
2. 出队时，要保证出队列的是最先进入队列的元素。栈1的元素挨个出栈（FILO，假设顺序为a-b）压进栈2 中（此时顺序为b-a），栈2再出栈（此时顺序为a-b）实现了FIFO
3. 需要注意的是，栈2为空时，栈1才可以将数压入其中，否则栈2的剩余元素比栈1弹出的元素先进，这样会导致栈1后进的元素先出。
4. 解决办法是当栈2为空的时候，要把栈1所有的元素全部出栈到栈2，之后栈2再出栈。

代码：

``` python
def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self):
        # return xx
        if self.stack2 == []:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
```

### 6. 旋转数组的最小数字

> *把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
> 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
> NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。*

解题思路：

1. 用二分法解决
2. 旋转后的数组就是两个有序的子数组，设定中间值mid是start+end//2
3. 整个数组中的最小值是在第二个子数组的左端点。将右端点作为target
4. 如果中间值mid>target，则说明最小值一定是在mid和target之间（不包含左端点mid）
5. 如果中间值mid<target，则说明最小值是在start和mid之间（包含端点）
6. 如果中间值mid=target，无法判断在哪一边，逐个比较。（eg. 1001111）

代码：

``` python
 def minNumberInRotateArray(self, rotateArray):
        # write code here
        if not rotateArray:
            return 0
        if len(rotateArray) == 1:
            return rotateArray[0]
        start = 0
        end = len(rotateArray)-1
        
        while rotateArray[start] >= rotateArray[end]:
            if end - start == 1:
                break
            mid = (start + end) // 2
            if rotateArray[start] == rotateArray[mid] == rotateArray[end]:
                temp = rotateArray[start]
                for i in range(start, end+1):
                    if temp > rotateArray[i]:
                        temp = rotateArray[i]
                        break
                return temp
            elif rotateArray[mid] > rotateArray[end]:
                start = mid
            elif rotateArray[mid] <= rotateArray[end]:
                end = mid
        return rotateArray[end]
```

### 7. 斐波那契数列

> *大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。n≤39*

解题思路：

1. 想法一：递归

   > 斐波那契数列公式：
   > $$
   > f[n] = f[n-1] + f[n-2]
   > $$
   > 其中：
   > $$
   > f[0]=0, f[1]=1
   > $$
   > 直接使用递归完成。
   >
   > 代码：
   >
   > ``` python
   > def Fibonacci(self, n):
   >     if n == 0 or n == 1:
   >         return n
   >     else:
   >         return Fibonacci(n-1)+Fibonacci(n-2)
   > ```
   >
   > 时间复杂度：O(2^n)	空间复杂度：递归栈的空间
   >
   > 缺点：慢，会超时

2. 想法二：

   > 在递归过程中其实存在很多重复计算，为了解决这个问题可以将算好的值保存下来。
   >
   > 1. 当n为0，1的时候直接返回n的值即可
   > 2. 当n大于1的时候，因为是按顺序计算，所以只需要保存前两个位置的数字即可。

代码实现：

``` python
def Fibonacci(self, n):
        # write code here 
        if n == 0 or n == 1:
            return n
        a = 0
        b = 1
        for i in range(2,n+1):
            c = a + b
            a = b
            b = c
        return c
```

### 8. 跳台阶

> *一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。*

解题思路：

1. 题目分析，假设f[i]表示在第i个台阶上可能的方法数。逆向思维。如果我从第n个台阶进行下台阶，下一步有2中可能，一种走到第n-1个台阶，一种是走到第n-2个台阶。所以公式为：

$$
f[n] = f[n-1] + f[n-2]
$$

2. 所以思路和斐波那契数列相同，但起始值变为:

$$
f[0]=1, f[1]=1
$$

代码：

``` python 
def jumpFloor(self, number):
        # write code here
        if number == 0 or number == 1:
            return 1
        a = 1
        b = 1
        for i in range(2, number+1):
            c = a + b
            a = b
            b = c
        return c
```

### 9. 变态跳台阶

> *一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。*

解题思路：

1. 同上一题，逆向思维解题可以得到跳上第n级台阶一共跳法有：

$$
f[n] = f[n-1] + f[n-2] + ... + f[0]
$$

2. 同理，跳第n-1级台阶的跳法为：

$$
f[n-1] = f[n-2] + f[n-3] + ... + f[0]
$$

3. 代入第一个式子可以得到：

$$
f[n] = 2*f[n-1]
$$

4. 起始条件为：
   $$
   f[0]=f[1]=1
   $$
   

代码：

``` python
def jumpFloorII(self, number):
        # write code here
        if number == 1 or number == 0:
            return 1
        a = 1
        for i in range(2, number+1):
            b = 2 * a 
            a = b 
        return b
```

### 10. 矩形覆盖

>*我们可以用2x1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2x1的小矩形无重叠地覆盖一个2xn的大矩形，总共有多少种方法？*
>
>*比如  n=3时，2x3的矩形块有3种覆盖方法：*
>
><img src="https://uploadfiles.nowcoder.com/images/20201028/59_1603852524038_7FBC41C976CACE07CB222C3B890A0995" alt="img" style="zoom:33%;" />

解题思路：

1. 找规律。

   | n    | 种数 |
   | ---- | ---- |
   | n=1  | 1    |
   | n=2  | 2    |
   | n=3  | 3    |
   | n=4  | 5    |

2. 斐波那契数列

代码：

``` python
def rectCover(self, number):
        # write code here
        if number == 0 or number == 1:
            return number
        a = 0
        b = 1
        for i in range(1, number+1):
            c = a + b
            a = b
            b = c
        return c
```

### 11. 二进制中1的个数

> *输入一个整数，输出该数32位二进制表示中1的个数。其中负数用补码表示。*

### 12. 数值的整数次方

> *给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。保证base和exponent不同时为0*

解题思路：

1. 分情况讨论

2. 先看指数exponent。如果是整数，则可以直接计算。如果是负数，则可以计算-exponent，也就是绝对值abs，之后返回1/result即可。

3. 计算方法，除了暴力累乘求解之外，可以使用二分法。

4. 当指数exponent为偶数时，e. g.
   $$
   2^8 = 2^4 * 2^4\\
   2^4 = 2^2 * 2^2\\
   2^2 = 2 * 2
$$
   
5. 当指数exponent为奇数时，e. g.
$$
   2^9 = 2^8 * 2^8
   \\2^8 = 2^4 * 2^4
   \\2^4 = 2^2 * 2^2
\\2^2 = 2 * 2
$$
   
6. 由分析可得，通过递归的思想：
$$
exponent为偶数： result = base ^  {exponent/2} * base ^  {exponent/2}\\
exponent为奇数： result = base ^  {exponent/2} * base ^  {exponent/2} * base
$$

7. 代码实现时分为两个部分，第一个部分定义函数mi，第二个部分为power直接调用函数运算幂。

8. mi函数：分为三种情况讨论。

   > 1. exponent为1，则直接返回base
   >
   > 2. exponent为偶数，则
   >    $$
   >    mi(base, exponent) = base ^  {exponent} 
   >    \\= base ^  {exponent/2} * base ^  {exponent/2}
   >\\= mi(base, exponent/2) * mi(base, exponent/2)
   >    $$
   >    
   > 3. exponent为奇数，则
   >    $$
   >    mi(base, exponent) = base ^  {exponent}
   >    \\
   >    = base ^  {exponent/2} * base ^  {exponent/2} * base
   >     \\
   >    = base ^  {(exponent/2)/2} * base ^  {(exponent/2)/2}*base
   >\\
   >    = mi(base, exponent/2)*mi(base, exponent/2)*base
   >   $$
   >    

9. power函数：分三种情况讨论。

   > 1. exponent为0，则直接返回1.
   > 2. exponent为负，取绝对值abs(exponent)，返回1/mi(base, abs(exponent))
   > 3. exponent为正，返回mi(base, exponent)


代码：

``` python
def Power(self, base, exponent):
        # write code here
        if exponent == 0:
            return 1
        if exponent < 0:
            return 1/self.mi(base, abs(exponent))
        else:
            return self.mi(base, exponent)
    def mi(self, base, exponent):
        if exponent == 1:
            return base
        if exponent%2: #奇数
            return self.mi(base, exponent/2) * self.mi(base, exponent/2) * base
        else:
            return self.mi(base, exponent/2) * self.mi(base, exponent/2)
```

### 13. 调整数组顺序使奇数位于偶数前面

> *输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。*

解题思路：

1. 想法一：使用辅助数组。使用两个辅助数组。当遇到奇数时将数据写入第一个数组，否则写入第二个数组。

   **代码：**

   ``` python
   def reOrderArray(self, array):
           # write code here
           odd = [] #奇
           even = [] #偶
           for i in array:
               if i%2 == 1:
                   odd.append(i)
               else:
                   even.append(i)
           return odd+even
   ```

2. 想法二：in-place算法。

   > 1. 初始化操作：设参数i表示将奇数放好的下一个位置。最开始的时候i==0，表示奇数还没有放好。j表示数组下表index， 初始值为0。
   > 2. 主要思想是遍历这个数组，找数组里面的奇数，将这个奇数移到最前面。
   > 3. 所以，当遇到偶数的时候就往后继续遍历，也就是j++
   > 4. 如果遇到一个奇数，index为j，此时应该将这个奇数插入到i的位置，也就是说[i, j-1]之间的所有数都要往后移动一位。也就是说将原来的array[i+1, j]赋值为array[i, j-1] 。
   > 5. 直到整个数组遍历结束。

   **代码：**

   ``` python
   
   ```

   

### 14. 链表中倒数第K个结点

> *输入一个链表，输出该链表中倒数第k个结点。*

解题思路：

1. 想法一：普通解法。倒数第k个就是正数第len-k-1个

   > 1. 首先计算整个链表的长度。循环计算链表长度，每next一个节点就len+1,当下一个结点也就是node.next为空的时候，node为最后一个结点，此时长度为len。
   > 2. 当k小于0或者k大于length的时候，表示不存在这样的数，返回null。
   > 3. 需要循环的次数是len-k次，也就是区间为[0, len-k]，每一次都获取node.next给node。循环结束后返回值为node。

代码：

``` python
def FindKthToTail(self, head, k):
        # write code here
        len = 0
        node = head
        while(node):
            len += 1
            node = node.next
        
        if k<0 or k>len:
            return None
        
        node = head
        for i in range(0, len-k):
            node = node.next
        return node
```

2. 想法二：双指针。快指针比慢指针先走k步，当快指针到达最后一个结点的时候，慢指针指向的就是倒数第k个结点。

   > 1. 在这个方法中不需要计算整个链表的长度，只需要判断快指针的下一个指针是否为空，为空的话则表示已经到了链表末尾了。
   > 2. 第一步需要让快指针先走k步。也就是先在[0, k]的区间中先循环赋值fast指针。
   > 3. 第二步，当fast结点不为空时，将快慢结点赋值为上一个node的下一个next，即fast/slow.next。
   > 4. 返回值应为slow

代码：

``` python
def FindKthToTail(self, head, k):
        # write code here
        fast = head
        for i in range(k):
            if fast == None:
                return None
            fast = fast.next
        
        slow = head
        while fast:
            slow = slow.next
            fast = fast.next
        return slow
```

### 15. 反转链表

> *输入一个链表，反转链表后，输出新链表的表头。*

解题思路：

1. 整体思路是调整链表指针来达到反转链表的目的。依次将指向后的指针断开，指向前一个数。
2. 需要三个指针。pre，cur，nex
3. pre用来指向已经反转好的链表的最后一个结点，也就是下一个待反转的结点需要指向的结点。刚开始的时候还没有开始反转，所以是空的。
4. cur用来指向待反转的链表的第一个结点，也就是现在在处理的那个结点。最开始的时候应该指向head。
5. nex用来指向待反转的链表的第二个结点，主要目的是存储后面没有被处理的链表。因为当断开cur与nex之间的指针时，cur之后的链表就丢失了，所以在断开之前要先用nex存储一下后面的链表。初始时应该是指向head.next
6. 接下来进行循环改变指针指向：
7. 第一步，保存后续链表。nex = cur.next
8. 第二步，将现在所在结点的指针指向pre，也就是将向后的指针指向前。cur.next = pre
9. 第三步，pre往后移动，pre = cur；cur往后移动，cur = nex
10. 开始下一轮循环，循环条件时cur!=None
11. 循环结束后，因为cur = nex，nex为空，cur也为空。返回pre，此时pre是反转后的头结点。

代码：

``` python
def ReverseList(self, pHead):
        # write code here
        pre = None
        cur = pHead
        
        while cur:
            nex = cur.next
            cur.next = pre
            pre = cur
            cur = nex
        return pre
```

### 16. 合并两个排序的链表

> *输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。*

 解题思路：

1. 新建一个链表，result = cur = ListNode(0)，cur作为用来移动比较的结点。浅拷贝，cur的变化会影响result的值。
2. 现有的两个链表为l1和l2。当l1和l2都不为空时开始循环。
3. 比较l1和l2的头结点的大小，也就是比较l1.val和l2.val的大小，如果l1小的话，则cur的下一个指向l1的这个结点，反之指向l2的结点。
4. 比如说，此时已经指向l1的头结点了，那么现在的l1就变成了该节点之后的链表，也就是l1.next。类似的，l2也可能变成l2.next。
5. 每次拼好一个新结点，cur都要往后移动。cur = cur.next
6. 一直循环上述步骤，直到l1或者l2为空的时候，现在我们得到的链表就是在某个链表已经比较完全之后的部分链表。
7. 现在还未合并完成的链表已经是排好序的了，所以将cur.next直接指向剩下的链表的头结点即可。
8. 因为并不知道剩下的链表是来自于那一条，所以直接指向l1orl2。

代码：

```python
    def Merge(self, pHead1, pHead2):
        # write code here
        result = cur = ListNode(0)
        while pHead1 and pHead2:
            if pHead1.val < pHead2.val:
                cur.next = pHead1
                pHead1 = pHead1.next
            else:
                cur.next = pHead2
                pHead2 = pHead2.next
            cur = cur.next
        cur.next = pHead1 or pHead2
        return result.next
```

### 17. 树的子结构

> *输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）*

涉及考点：TreeNode

> TreeNode是经典的二叉树节点，在数据的序列化和反序列按照层遍历来处理的。
>
> <img src=" https://uploadfiles.nowcoder.com/images/20200718/68_1595063284833_17B9378350009B5D8CD4F47029FB7EA8">
>
> 以上二叉树会被序列化为 {1,2,3,#,#,4,#,#,5}
> 1：root节点1，是第一层
> 2,3：然后第二层是2，3
> \#,#,4,#：第三层分别是2节点的两个孩子节点空，用#来表示，然后3节点的左孩子为4，右孩子节点为#
> \#,5：第四层4节点的左孩子是空，右孩子为5
> 最后一层5节点的两个空孩子不便利

解题思路：

1. 递归求解。参考递归三部曲。

   > 1. 判断递归函数的功能：判断两个数是否有相同的结构，如果相同，返回true，否则返回false
   > 2. 判断停止递归的条件：如果树B是空树，则直接返回true。如果树A是空树，则直接返回false。
   > 3. 确定递归参数：
   >    1. 如果根节点不相同，则直接返回false
   >    2. 如果相等的话，就接着判断左右子树是否相等      
   >    3. 当左子树、右子树、根节点均相等时返回值为true。                                                                                                                                               

2. 接下来是递归函数的具体内容。

   > 1. 将树A的每个节点作为根节点与B来比较。遍历A的节点，可以采用先序遍历。
   >    1. 