// Advanced Navigation Bar Effects for JoeyLLM Website

document.addEventListener('DOMContentLoaded', function() {
    // Reference to navbar elements
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelectorAll('.nav-links a');
    const menuToggle = document.querySelector('.menu-toggle');
    
    // Apply different effects to different nav links
    navLinks.forEach((link, index) => {
        // Apply different effects based on index
        switch(index % 5) {
            case 0:
                // Add gradient underline effect
                link.classList.add('gradient-underline');
                break;
            case 1:
                // Add pill effect
                link.classList.add('pill-effect');
                break;
            case 2:
                // Add scale effect
                link.classList.add('scale-effect');
                break;
            case 3:
                // Add Australian flag effect
                link.classList.add('aussie-flag-link');
                break;
            case 4:
                // Add bounce effect on click
                link.addEventListener('click', function() {
                    this.classList.add('bounce');
                    setTimeout(() => {
                        this.classList.remove('bounce');
                    }, 500);
                });
                break;
        }
        
        // Set active link based on current scroll position
        link.addEventListener('click', function() {
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Add active class to current section based on scroll position
    window.addEventListener('scroll', function() {
        // Navbar scroll effect
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        // Highlight active section in navbar
        const scrollPosition = window.scrollY + 100;
        
        document.querySelectorAll('section').forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    });
    
    // Mobile menu toggle animation
    menuToggle.addEventListener('click', function() {
        this.classList.toggle('active');
    });
    
    // Add theme toggle functionality
    const themeToggleHtml = `
        <button class="theme-toggle" aria-label="Toggle dark mode">
            <i class="theme-icon">🌓</i>
        </button>
    `;
    
    // Insert theme toggle button into navbar
    const navContainer = document.querySelector('.navbar-container');
    const themeToggleContainer = document.createElement('div');
    themeToggleContainer.innerHTML = themeToggleHtml;
    navContainer.appendChild(themeToggleContainer);
    
    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle');
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-theme');
        const icon = this.querySelector('.theme-icon');
        if (document.body.classList.contains('dark-theme')) {
            icon.textContent = '☀️';
        } else {
            icon.textContent = '🌓';
        }
    });
    
    // Add notification badge to a random nav item for demo
    const randomNavItem = navLinks[Math.floor(Math.random() * navLinks.length)];
    randomNavItem.parentElement.classList.add('nav-item-with-badge');
    const badge = document.createElement('span');
    badge.classList.add('badge');
    badge.textContent = '1';
    randomNavItem.parentElement.appendChild(badge);
    
    // Create a search bar
    const searchHtml = `
        <div class="search-container">
            <input type="text" class="search-input" placeholder="Search...">
            <button class="search-btn">🔍</button>
        </div>
    `;
    
    // Insert search bar into navbar
    const searchContainer = document.createElement('div');
    searchContainer.innerHTML = searchHtml;
    navContainer.insertBefore(searchContainer, themeToggleContainer);
    
    // Create a dropdown menu for demo
    const createDropdown = () => {
        // Select a nav item to convert to dropdown
        const targetNavItem = document.querySelector('.nav-links li:nth-child(4)');
        targetNavItem.classList.add('dropdown');
        
        const link = targetNavItem.querySelector('a');
        const linkText = link.textContent;
        
        // Create dropdown content
        const dropdownContent = document.createElement('div');
        dropdownContent.classList.add('dropdown-content');
        
        // Add dropdown items
        const dropdownItems = [
            'Architecture', 
            'Training', 
            'Datasets', 
            'API'
        ];
        
        dropdownItems.forEach(item => {
            const dropdownLink = document.createElement('a');
            dropdownLink.href = '#';
            dropdownLink.textContent = item;
            dropdownContent.appendChild(dropdownLink);
        });
        
        // Add dropdown to nav item
        targetNavItem.appendChild(dropdownContent);
    };
    
    // Initialize dropdown
    createDropdown();
    
    // Add scroll reveal animation to sections
    const revealSections = () => {
        const sections = document.querySelectorAll('section');
        
        const revealSection = (entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = 1;
                    entry.target.style.transform = 'translateY(0)';
                    observer.unobserve(entry.target);
                }
            });
        };
        
        const sectionObserver = new IntersectionObserver(revealSection, {
            root: null,
            threshold: 0.15
        });
        
        sections.forEach(section => {
            section.style.opacity = 0;
            section.style.transform = 'translateY(50px)';
            section.style.transition = 'all 0.5s ease';
            sectionObserver.observe(section);
        });
    };
    
    // Initialize section reveal animations
    revealSections();
});
